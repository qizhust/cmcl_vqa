import torch
import torch.nn as nn

from modules import heads, objectives
from modules.pre_utils import set_task, set_metrics
from modules.optim_utils import init_weights
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
import modules.vision_transformers as vit

class ViLTTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        set_metrics(self, config.loss_names)
        self.current_tasks = set_task(config.loss_names)
        self.max_image_len = getattr(config, "max_image_len", -1)

        # build text encoder
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(init_weights)

        if config.resume_from == "":
            self.transformer = getattr(vit, config.vit)(
                pretrained=True, config=config
            )
        else:
            self.transformer = getattr(vit, config.vit)(
                pretrained=False, config=config
            )

        self.pooler = heads.Pooler(config.hidden_size)
        self.pooler.apply(init_weights)

        # task specific heads
        if config.loss_names["vqa"] > 0:
            self.vqa_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                nn.LayerNorm(config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.vqav2_label_size),
            )
            self.vqa_classifier.apply(init_weights)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.max_image_len,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        vqa_logits = self.vqa_classifier(cls_feats)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "vqa_logits": vqa_logits
        }

        return ret

    def forward(self, batch):
        ret = {'non_scalars': {}, 'scalars': {}}
        if len(self.current_tasks) == 0:  # for test_only
            ret['non_scalars'].update(self.infer(batch))
            return ret

        for ele in self.current_tasks:
            non_scalars, scalars = getattr(objectives, 'compute_'+ele)(self, batch)

        ret['non_scalars'].update(non_scalars)
        ret['scalars'].update(scalars)
        return ret

