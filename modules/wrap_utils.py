import os
import glob
import json
import torch.distributed as dist


def epoch_wrapup(model, current_tasks):
    phase = "train" if model.training else "val"
    the_metric = 0

    ret = {}
    for task in current_tasks:
        value = 0

        if task == "vqa":
            value = getattr(model, f"{phase}_{task}_score").compute()
            ret.update({f"{task}/{phase}/score_epoch": value})
            getattr(model, f"{phase}_{task}_score").reset()
            ret.update(
                {f"{task}/{phase}/loss_epoch":
                getattr(model, f"{phase}_{task}_loss").compute()}
            )
            getattr(model, f"{phase}_{task}_loss").reset()
        elif task == "itm":
            value = getattr(model, f"{phase}_{task}_accuracy").compute()
            ret.update({f"{task}/{phase}/accuracy_epoch": value})
            getattr(model, f"{phase}_{task}_accuracy").reset()
            ret.update(
                {f"{task}/{phase}/loss_epoch":
                getattr(model, f"{phase}_{task}_loss").compute()}
            )
            getattr(model, f"{phase}_{task}_loss").reset()
        else:
            value = getattr(model, f"{phase}_{task}_accuracy").compute()
            ret.update({f"{task}/{phase}/accuracy_epoch": value})
            getattr(model, f"{phase}_{task}_accuracy").reset()
            ret.update(
                {f"{task}/{phase}/loss_epoch":
                getattr(model, f"{phase}_{task}_loss").compute()}
            )
            getattr(model, f"{phase}_{task}_loss").reset()

        the_metric += value

    ret.update({f"{phase}/the_metric": the_metric})
    return ret


def vqa_test_step(batch, output, id2answer, if_gqa=False):
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": if_gqa}


def vqa_test_wrapup(outs, dst_path, global_rank):
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})

    with open(f"{dst_path}/vqa_submit_{global_rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    dist.barrier()

    if global_rank == 0:
        jsons = list()
        paths = list(glob.glob(f"{dst_path}/vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        with open(f"{dst_path}/vqa_submit.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    dist.barrier()
    os.remove(f"{dst_path}/vqa_submit_{global_rank}.json")
