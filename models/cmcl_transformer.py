import torch
import torch.nn as nn

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from modules.bert_model import BertCrossLayer
from modules import heads, objectives
from modules.encoder_utils import build_text_encoder, build_vision_encoder
from modules.pre_utils import set_task, set_metrics
from modules.optim_utils import init_weights


@torch.no_grad()
def build_connected_component(image_base):
    image_sim = torch.matmul(image_base, image_base.transpose(0,1))
    image_norm = (image_base*image_base).sum(-1).sqrt().unsqueeze(-1)
    image_norm = torch.matmul(image_norm, image_norm.transpose(0,1))
    dist = image_sim/image_norm  # here dist means normalized similarity

    device = dist.device
    b = dist.size(0)
    dist = dist - torch.eye(b, b, device=device) * 2
    x = torch.arange(b, device=device).unsqueeze(1).repeat(1,1).flatten()
    y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
    rx = torch.cat([x, y]).cpu().numpy()
    ry = torch.cat([y, x]).cpu().numpy()
    v = np.ones(rx.shape[0])
    graph = csr_matrix((v, (rx, ry)), shape=(b,b))
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    labels = torch.tensor(labels, device=device)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
    return mask


class CMCLTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        set_metrics(self, config.loss_names)
        self.current_tasks = set_task(config.loss_names)
        self.cl_type = "cl_nce_graph" if "nce_graph" in config.cl_type else "cl_nce"
        self.cl_loss_type = "cl_loss_multipos" if "multipos" in config.cl_type else "cl_loss"
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

            self.ans_embed = nn.Embedding(config.vqav2_label_size, config.answer_hidden_size)
            self.ans_embed.apply(init_weights)
            self.qa_encoder = nn.Linear(config.hidden_size+config.answer_hidden_size, config.hidden_size)
            self.qa_encoder.apply(init_weights)

    def cl_nce(self, batch, cls_feats_image_base, text_embeds, device):
        qa_encoded = self._get_qa_encoding(batch, text_embeds, device)
        
        f_xc = {}
        for ele in ['image', 'false_image']:
            f_xc[ele] = torch.matmul(cls_feats_image_base[ele], qa_encoded.transpose(0, 1)).exp()

        f_xc_all = torch.cat([f_xc['image'], f_xc['false_image']], 0)  # col for c, row for im
        f_xc_all = torch.diagonal(f_xc['image'])/f_xc_all.sum(0)
        loss_infonce = -(f_xc_all.log() * (1.0 - torch.tensor(batch['yes_type']).to(device))).sum()
        return loss_infonce

    def _get_qa_encoding(self, batch, text_embeds, device):
        # get cls text embedding before multi-modal fusing
        cls_feats_text_base = self.cross_modal_text_pooler(text_embeds)

        ans_labels = []
        for ans, scores in zip(batch['vqa_labels'], batch['vqa_scores']):
            highest_idx = sorted(range(len(ans)), key=lambda k: scores[k])[-1]
            ans_labels.append(ans[highest_idx])  # use the answer with the highest score
        ans_labels = torch.LongTensor(ans_labels).to(device)
        ans_emb = self.ans_embed(ans_labels)
        qa_encoded = self.qa_encoder(torch.cat([cls_feats_text_base, ans_emb], -1))
        return qa_encoded

    def _cl_loss(self, f_xc, neg_mask, sample_size, yes_type, device):
        f_xc_all = torch.cat([f_xc['image'], f_xc['false_image']], 0)  # col for c, row for im
        eye_mask = torch.cat([torch.eye(sample_size), torch.zeros(sample_size, sample_size)], 1).to(device)
        sample_mask = ~neg_mask[:sample_size] + eye_mask  # the negative samples outside the graph and the positive sample
        f_xc_all = torch.diagonal(f_xc['image'])/(f_xc_all * sample_mask.T).sum(0)
        loss_infonce = -(f_xc_all.log() * (1.0 - torch.tensor(yes_type).to(device))).sum()
        return loss_infonce

    def _cl_loss_multipos(self, f_xc, neg_mask, sample_size, yes_type, device):
        f_xc_all = torch.cat([f_xc['image'], f_xc['false_image']], 0)  # col for c, row for im
        loss_infonce = -(f_xc_all / f_xc_all.sum(0)).log() * neg_mask[:sample_size].T
        loss_infonce = loss_infonce.sum(0)/neg_mask[:sample_size].sum(1)
        loss_infonce = (loss_infonce * (1.0 - torch.tensor(yes_type).to(device))).sum()
        return loss_infonce

    def cl_nce_graph(self, batch, cls_feats_image_base, text_embeds, device):
        image_base = torch.cat([cls_feats_image_base['image'], cls_feats_image_base['false_image']], 0)
        neg_mask = build_connected_component(image_base)  # False for selected negative sample
        sample_size = batch['text_masks'].size(0)

        qa_encoded = self._get_qa_encoding(batch, text_embeds, device)

        # calculate InfoNCE loss
        f_xc = {}
        for ele in ['image', 'false_image']:
            f_xc[ele] = torch.matmul(cls_feats_image_base[ele], qa_encoded.transpose(0, 1)).exp()

        loss_infonce = getattr(self, '_'+self.cl_loss_type)(f_xc, neg_mask, sample_size, batch['yes_type'], device)

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

        image_embeds, image_masks, extend_image_masks, cls_feats_image_base, = {}, {}, {}, {}
        # get image embeds from vision module and cross_modal_image module
        img_list = ['image', 'false_image'] if self.training else ['image']
        for ele in img_list:
            image_embeds[ele] = self.vision_encoder(batch[ele][0])
            image_embeds[ele] = self.cross_modal_image_transform(image_embeds[ele])
            image_masks[ele] = torch.ones((image_embeds[ele].size(0), image_embeds[ele].size(1)), dtype=torch.long, device=device)
            extend_image_masks[ele] = self.text_encoder.get_extended_attention_mask(image_masks[ele], image_masks[ele].size(), device)

            image_embeds[ele] = image_embeds[ele] + self.token_type_embeddings(torch.full_like(image_masks[ele], image_token_type_idx))
            cls_feats_image_base[ele] = self.cross_modal_image_pooler(self.vision_encoder.fetch_pool_data(image_embeds[ele]))

        # calculate contrastive loss
        if self.training:
            loss_infonce = getattr(self, self.cl_type)(batch, cls_feats_image_base, text_embeds, device)
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
