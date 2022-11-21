import torch
import torch.nn as nn

from modules import heads, objectives
from modules.pre_utils import set_task, set_metrics
from modules.optim_utils import init_weights
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
import modules.vision_transformers as vit


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


class CMCLViLT(nn.Module):
    def __init__(self, config):
        super().__init__()

        set_metrics(self, config.loss_names)
        self.current_tasks = set_task(config.loss_names)
        self.max_image_len = getattr(config, "max_image_len", -1)
        self.cl_type = "cl_nce_graph" if "nce_graph" in config.cl_type else "cl_nce"
        self.cl_loss_type = "cl_loss_multipos" if "multipos" in config.cl_type else "cl_loss"
        self.nce_lambda = config.nce_lambda

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

            self.ans_embed = nn.Embedding(config.vqav2_label_size, config.answer_hidden_size)
            self.ans_embed.apply(init_weights)
            self.qa_encoder = nn.Linear(config.hidden_size+config.answer_hidden_size, config.hidden_size)
            self.qa_encoder.apply(init_weights)

    def cl_nce(self, batch, image_embeds, text_embeds, device):
        qa_encoded = self._get_qa_encoding(batch, text_embeds, device)
        
        f_xc = {}
        for ele in ['image', 'false_image']:
            f_xc[ele] = torch.matmul(image_embeds[ele][:, 0], qa_encoded.transpose(0, 1)).exp()

        f_xc_all = torch.cat([f_xc['image'], f_xc['false_image']], 0)  # col for c, row for im
        f_xc_all = torch.diagonal(f_xc['image'])/f_xc_all.sum(0)
        loss_infonce = -(f_xc_all.log() * (1.0 - torch.tensor(batch['yes_type']).to(device))).sum()
        return loss_infonce

    def _get_qa_encoding(self, batch, text_embeds, device):
        ans_labels = []
        for ans, scores in zip(batch['vqa_labels'], batch['vqa_scores']):
            highest_idx = sorted(range(len(ans)), key=lambda k: scores[k])[-1]
            ans_labels.append(ans[highest_idx])  # use the answer with the highest score
        ans_labels = torch.LongTensor(ans_labels).to(device)
        ans_emb = self.ans_embed(ans_labels)
        qa_encoded = self.qa_encoder(torch.cat([text_embeds, ans_emb], -1))
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

    def cl_nce_graph(self, batch, image_embeds, text_embeds, device):
        image_base = torch.cat([self.pooler(image_embeds['image']), self.pooler(image_embeds['false_image'])], 0)
        neg_mask = build_connected_component(image_base)  # False for selected negative sample
        sample_size = batch['text_masks'].size(0)

        qa_encoded = self._get_qa_encoding(batch, text_embeds, device)

        # calculate InfoNCE loss
        f_xc = {}
        for ele in ['image', 'false_image']:
            f_xc[ele] = torch.matmul(image_embeds[ele][:, 0], qa_encoded.transpose(0, 1)).exp()

        loss_infonce = getattr(self, '_'+self.cl_loss_type)(f_xc, neg_mask, sample_size, batch['yes_type'], device)

        return loss_infonce

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        device = batch['text_ids'].device

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))  # [B, L1, D]

        image_embeds, image_masks, patch_index, image_labels = {}, {}, {}, {}
        img_list = ['image', 'false_image'] if self.training else ['image']
        for ele in img_list:
            image_embeds[ele], image_masks[ele], patch_index[ele], image_labels[ele] = self.transformer.visual_embed(
                batch[ele][0], max_image_len=self.max_image_len, mask_it=mask_image)
            image_embeds[ele] = image_embeds[ele] + self.token_type_embeddings(
                torch.full_like(image_masks[ele], image_token_type_idx))  # [B, L2, D]

        if self.training:
            loss_infonce = getattr(self, self.cl_type)(batch, image_embeds, self.pooler(text_embeds), device)
        else:
            loss_infonce = 0.

        co_embeds = torch.cat([text_embeds, image_embeds['image']], dim=1)  # [B, L1+L2, D]
        co_masks = torch.cat([text_masks, image_masks['image']], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )  # [B, L1, D], [B, L2, D]
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
            "vqa_logits": vqa_logits,
            "loss_infonce": loss_infonce
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

