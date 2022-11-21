import torch
import torch.nn as nn

# from modules.bert_model import BertLMHeadModel
from modules import heads, objectives
from modules.albef_vit import VisionTransformer
from modules.albef_xbert import BertConfig, BertModel
from modules.pre_utils import set_task, set_metrics
from modules.optim_utils import init_weights
from functools import partial


class ALBEF(nn.Module):
    def __init__(self, config):
        super().__init__()

        set_metrics(self, config.loss_names)
        self.current_tasks = set_task(config.loss_names)
        self.distill = config.distill

        self.visual_encoder = VisionTransformer(
            img_size=config.image_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 

        # build text encoder
        bert_config = BertConfig.from_json_file('cfgs/config_bert.json')
        self.text_encoder = BertModel.from_pretrained(config.tokenizer, config=bert_config, add_pooling_layer=False) 
        # self.text_decoder = BertLMHeadModel.from_pretrained(config.tokenizer, config=bert_config)    

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config.image_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))             
            self.text_encoder_m = BertModel.from_pretrained(config.tokenizer, config=bert_config, add_pooling_layer=False)   
            self.text_decoder_m = BertLMHeadModel.from_pretrained(config.tokenizer, config=bert_config)   
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            self.copy_params() 
            self.momentum = 0.995

        # task specific heads
        if config.loss_names["vqa"] > 0:
            self.vqa_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                nn.LayerNorm(config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.vqav2_label_size),
            )
            self.vqa_classifier.apply(init_weights)

    def infer(self, batch, mask_text=False, mask_image=False, image_token_type_idx=1, img=None):
        device = batch['text_ids'].device
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0].to(device)

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        image_embeds = self.visual_encoder(img)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)

        question_output = self.text_encoder(input_ids=text_ids,
                                            attention_mask=text_masks,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_masks,
                                            return_dict=True)

        vqa_logits = self.vqa_classifier(question_output['last_hidden_state'][:, 0])

        ret = {"text_labels": text_labels,
               "text_ids": text_ids,
               "text_masks": text_masks,
               "vqa_logits": vqa_logits}

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

