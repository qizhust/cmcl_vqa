import os
import json
import argparse

from .metrics import Accuracy, Scalar, VQAScore

def save_hyperparameters(config, save_dir):
    with open(os.path.join(save_dir, 'hparams.json'), 'w') as fd:
        json.dump(config.__dict__, fd)


def set_metrics(model, loss_names):
    for split in ["train", "val"]:
        for k, v in loss_names.items():
            if v <= 0:
                continue
            if k == "vqa":
                setattr(model, f"{split}_vqa_score", VQAScore())
                setattr(model, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(model, f"{split}_{k}_accuracy", Accuracy())
                setattr(model, f"{split}_{k}_loss", Scalar())
            else:
                setattr(model, f"{split}_{k}_accuracy", Accuracy())
                setattr(model, f"{split}_{k}_loss", Scalar())


def set_task(loss_names):
    current_tasks = [
        k for k, v in loss_names.items() if v > 0
    ]
    return current_tasks

def logging_scalars(global_rank, data, phase, writer, step):
    if global_rank == 0:
        for k, v in data.items():
            writer.add_scalar(f"{phase}/{k}", v.item(), step)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value
