import numpy as np
import functools
from collections import defaultdict

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (DataCollatorForLanguageModeling, DataCollatorForWholeWordMask)

from .vqav2_dataset import VQAv2Dataset
from .vqacp_dataset import VQACPDataset

ds_dict = {"vqav2": VQAv2Dataset, "vqacp": VQACPDataset}


def build_dataloader(cfg, split):
    transform_key = 'train_transform_keys' if 'train' in split else 'val_transform_keys'
    # drop_last = False if 'test' in split else True
    drop_last = False
    dataset = ds_dict[cfg.dataset_name](cfg.data_root, getattr(cfg, transform_key), 
                           cfg.image_size, cfg.tokenizer, split=split, 
                           draw_false_image_vqa=cfg.draw_false_image_vqa)

    collator = (DataCollatorForWholeWordMask if cfg.whole_word_masking
                else DataCollatorForLanguageModeling)
    mlm_collator = collator(tokenizer=dataset.tokenizer, mlm=True, mlm_probability=cfg.mlm_prob)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, drop_last=drop_last,
                            batch_size=cfg.per_gpu_batchsize, num_workers=cfg.num_workers,
                            collate_fn=functools.partial(dataset.collate, mlm_collator=mlm_collator))
    return dataloader, dataset, sampler


def build_vqa_id2answer(cfg):
    dataset = {}
    for split in ['train', 'val']:
        dataset[split] = ds_dict[cfg.dataset_name](cfg.data_root, getattr(cfg, split+'_transform_keys'), 
                                      cfg.image_size, cfg.tokenizer, split=split)
        dataset[split].open_arrow()
    train_answers = dataset['train'].table["answers"].to_pandas().tolist()
    val_answers = dataset['val'].table["answers"].to_pandas().tolist()
    train_labels = dataset['train'].table["answer_labels"].to_pandas().tolist()
    val_labels = dataset['val'].table["answer_labels"].to_pandas().tolist()

    all_answers = [c for c in train_answers + val_answers if c is not None]
    all_answers = [l for lll in all_answers for ll in lll for l in ll]
    all_labels = [c for c in train_labels + val_labels if c is not None]
    all_labels = [l for lll in all_labels for ll in lll for l in ll]

    answer2id = {k: v for k, v in zip(all_answers, all_labels)}
    sorted_a2i = sorted(answer2id.items(), key=lambda x: x[1])
    
    id2answer = defaultdict(lambda: "unknown")
    for k, v in sorted_a2i:
        id2answer[v] = k

    return id2answer


def fetch_answer_categories(cfg):
    id2answer = build_vqa_id2answer(cfg)

    labels = np.zeros((3, cfg.vqav2_label_size), dtype='float32')
    for k, v in id2answer.items():
        if v in ['yes', 'no']:
            labels[0, k] = 1
        else:
            try:
                num = int(v)
                labels[1, k] = 1
            except:
                labels[2, k] = 1

    labels = labels/labels.sum(-1, keepdims=True)
    return labels
