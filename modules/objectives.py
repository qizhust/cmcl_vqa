import torch
import torch.nn.functional as F

def compute_mlm(model, batch):
    infer = model.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = model.mlm_score(infer["text_feats"])
    vocab_size = mlm_logits.size(-1)
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, vocab_size),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    scalars = {"mlm_loss": mlm_loss}

    ret = {
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if model.training else "val"
    loss = getattr(model, f"{phase}_mlm_loss")(ret['scalars']["mlm_loss"])
    acc = getattr(model, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )

    scalars["mlm_acc"] = acc

    return ret, scalars

def compute_itm(model, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        batch["text_labels"].device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = model.infer(batch, mask_text=False, mask_image=False)

    itm_logits = model.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    scalars = {"itm_loss": itm_loss}

    ret = {
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if model.training else "val"
    loss = getattr(model, f"{phase}_itm_loss")(scalars["itm_loss"])
    acc = getattr(model, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )

    scalars["itm_acc"] = acc

    return ret, scalars

def compute_vqa(model, batch):
    infer = model.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = infer["vqa_logits"]
    vqav2_label_size = vqa_logits.size(-1)
    vqa_targets = torch.zeros(
        len(vqa_logits), vqav2_label_size
    ).to(infer["vqa_logits"].device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    vqa_loss = vqa_loss + getattr(model, 'nce_lambda', 0.) * infer.get('loss_infonce', 0.)

    if 'qa_logits' in infer:
        qa_loss = F.binary_cross_entropy_with_logits(infer['qa_logits'], vqa_targets) * vqa_targets.shape[1]
        vqa_loss += qa_loss

    scalars = {"vqa_loss": vqa_loss}

    ret = {
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if model.training else "val"
    loss = getattr(model, f"{phase}_vqa_loss")(scalars["vqa_loss"])
    score = getattr(model, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )

    scalars["vqa_score"] = score

    return ret, scalars
