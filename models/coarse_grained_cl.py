import torch
import torch.nn as nn

from modules.bert_model import BertCrossLayer
from modules import heads, objectives
from modules.encoder_utils import build_text_encoder, build_vision_encoder
from modules.pre_utils import set_task, set_metrics
from modules.optim_utils import init_weights


class CoarseCL(nn.Module):
    def __init__(self, config):
        super().__init__()

        set_metrics(self, config.loss_names)
        self.current_tasks = set_task(config.loss_names)
        self.cl_type = "cl_nce"
        self.nce_lambda = config.nce_lambda

        # build text encoder
        bert_config, self.text_encoder = build_text_encoder(config)

        # build vision encoder
        self.vision_encoder = build_vision_encoder(config)

        # set requires_grad=False to unused parameters
        self.text_encoder.pooler.dense.weight.requires_grad = False
        self.text_encoder.pooler.dense.bias.requires_grad = False
        self.vision_encoder.vit.positional_embedding.requires_grad = False
        self.vision_encoder.vit.ln_final.weight.requires_grad = False
        self.vision_encoder.vit.ln_final.bias.requires_grad = False

        # build cross modal projection layer, pre-fusion
        self.cross_modal_text_transform = nn.Linear(config.input_text_embed_size, config.hidden_size)
        self.cross_modal_text_transform.apply(init_weights)
        self.cross_modal_image_transform = nn.Linear(config.input_image_embed_size, config.hidden_size)
        self.cross_modal_image_transform.apply(init_weights)

        # two types of embedding
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(init_weights)

        # build cross modal fusion layers
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config.num_top_layer)])
        self.cross_modal_image_layers.apply(init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config.num_top_layer)])
        self.cross_modal_text_layers.apply(init_weights)

        # build top pooler layers for global semantics
        self.cross_modal_image_pooler = heads.Pooler(config.hidden_size)
        self.cross_modal_image_pooler.apply(init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config.hidden_size)
        self.cross_modal_text_pooler.apply(init_weights)

        # task specific heads
        if config.loss_names["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(init_weights)

        if config.loss_names["itm"] > 0:
            self.itm_score = heads.ITMHead(config.hidden_size*2)
            self.itm_score.apply(init_weights)

        if config.loss_names["vqa"] > 0:
            self.vqa_classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.LayerNorm(config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.vqav2_label_size),
            )
            self.vqa_classifier.apply(init_weights)

    def cl_nce(self, batch, image_embeds, text_embeds, device):
        batch_size = len(batch['vqa_answer'])
        pos_indicator = torch.zeros((batch_size, batch_size)).to(device)
        for i, ref in enumerate(batch['vqa_answer'][:-1]):
            for j, candi in enumerate(batch['vqa_answer'][i+1:], start=i+1):
                if candi[0] == ref[0]:
                    pos_indicator[i, j] = 1
                    pos_indicator[j, i] = 1

        if pos_indicator.sum() > 0:
            cls_feats_image_base = self.cross_modal_image_pooler(self.vision_encoder.fetch_pool_data(image_embeds))
            cls_feats_text_base = self.cross_modal_text_pooler(text_embeds)
            IQ = cls_feats_image_base * cls_feats_text_base

            mult_mat = (IQ @ IQ.transpose(0, 1)).exp()

            eye_mat = torch.eye(batch_size).to(device)
            per_cl = ((mult_mat * pos_indicator).sum(1) + 1e-6) / (mult_mat * (1-eye_mat)).sum(1)
            loss_infonce = -(per_cl.log() * pos_indicator.sum(0)).sum() / batch_size
        else:
            loss_infonce = 0.0
        return loss_infonce

    def infer(self, batch, mask_text=False, mask_image=False, image_token_type_idx=1, img=None):
        device = batch['text_ids'].device

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        # encode text
        text_embeds = self.text_encoder.embeddings(input_ids=text_ids)
        input_shape = text_masks.size()
        extend_text_masks = self.text_encoder.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_encoder.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        image_embeds, image_masks, extend_image_masks = {}, {}, {}
        # get image embeds from vision module and cross_modal_image module
        img_list = ['image']
        for ele in img_list:
            image_embeds[ele] = self.vision_encoder(batch[ele][0])
            image_embeds[ele] = self.cross_modal_image_transform(image_embeds[ele])
            image_masks[ele] = torch.ones((image_embeds[ele].size(0), image_embeds[ele].size(1)), dtype=torch.long, device=device)
            extend_image_masks[ele] = self.text_encoder.get_extended_attention_mask(image_masks[ele], image_masks[ele].size(), device)

            image_embeds[ele] = image_embeds[ele] + self.token_type_embeddings(torch.full_like(image_masks[ele], image_token_type_idx))

        # calculate contrastive loss
        if self.training:
            loss_infonce = getattr(self, self.cl_type)(batch, image_embeds['image'], text_embeds, device)
        else:
            loss_infonce = 0.

        # multi-modal fusion
        x, y = text_embeds, image_embeds['image']
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks['image'])
            y1 = image_layer(y, x, extend_image_masks['image'], extend_text_masks)
            x, y = x1[0], y1[0]

        # calculate global representation
        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        cls_feats_image = self.cross_modal_image_pooler(self.vision_encoder.fetch_pool_data(image_feats))
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        vqa_logits = self.vqa_classifier(cls_feats)

        ret = {
            "text_feats": text_feats, 
            "image_feats": image_feats,
            "cls_feats": cls_feats, 
            "text_labels": text_labels,
            "text_ids": text_ids, 
            "text_masks": text_masks,
            "loss_infonce": loss_infonce,
            "vqa_logits": vqa_logits
        }

        return ret

    def forward(self, batch):
        if len(self.current_tasks) == 0:
            return self.infer(batch)

        ret = {'non_scalars': {}, 'scalars': {}}
        for ele in self.current_tasks:
            non_scalars, scalars = getattr(objectives, 'compute_'+ele)(self, batch)

        ret['non_scalars'].update(non_scalars)
        ret['scalars'].update(scalars)
        return ret
