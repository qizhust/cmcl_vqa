import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .meter_transformer import METERTransformer
from .cmcl_transformer import CMCLTransformer
from .coarse_grained_cl import CoarseCL
from .vilt import ViLTTransformer
from .cmcl_vilt import CMCLViLT
from .cmcl_transk import CMCLTransformerk
from .albef import ALBEF
from .cmcl_albef import CMCLAlbef

from modules.albef_vit import interpolate_pos_embed


model_dict = {'meter': METERTransformer, 
              'cmcl': CMCLTransformer, 
              'coarsecl': CoarseCL,
              'vilt': ViLTTransformer,
              'cmcl_vilt': CMCLViLT,
              'cmcl_topk': CMCLTransformerk,
              'albef': ALBEF,
              'cmcl_albef': CMCLAlbef}

def build_model(cfg, gpu_id, is_distributed=True):
    model = model_dict[cfg.model_type](cfg).cuda(gpu_id)
    if is_distributed:
        model = DDP(model, device_ids=[gpu_id], output_device=gpu_id)
    return model

def load_ckpt(model, cfg, load_path):
    ckpt = torch.load(load_path, map_location="cpu")
    if cfg.load_pretrained_meter==3:  # load from vilt
        load_albef_ckpt(model, ckpt['model'])
        return {"state_dict": ckpt['model']}    
    elif cfg.load_pretrained_meter==2:  # load from vilt
        load_meter_ckpt(model, ckpt['state_dict'])  
        return {"state_dict": ckpt['state_dict']}
    elif cfg.load_pretrained_meter==1:  # load from meter
        state_dict = model.module.vision_encoder.load_checkpoints(ckpt['state_dict'], cfg)
        load_meter_ckpt(model, state_dict)
        return {"state_dict": ckpt['state_dict']}
    elif cfg.load_pretrained_meter==0:  # for meter inference only
        state_dict = model.module.vision_encoder.load_checkpoints(ckpt['state_dict'], cfg)
        model.module.load_state_dict(state_dict)  
        return ckpt
    elif cfg.load_pretrained_meter==-1:
        model.module.load_state_dict(ckpt['state_dict'])  
        return ckpt

def load_meter_ckpt(model, state_dict):
    new_state_dict = {}
    for k, v in model.state_dict().items():
        if 'vision_encoder' in k:
            tmp = 'vit_model.' + '.'.join(k.split('.')[3:])
            new_state_dict[k] = state_dict[tmp]
        elif 'text_encoder' in k:
            tmp = 'text_transformer.' + '.'.join(k.split('.')[2:])
            new_state_dict[k] = state_dict[tmp]
        else:
            new_state_dict[k] = state_dict.get('.'.join(k.split('.')[1:]), v)
    model.load_state_dict(new_state_dict, strict=False)

def load_albef_ckpt(model, state_dict):
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.module.visual_encoder)  
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.','')         
            state_dict[encoder_key] = state_dict[key] 
        # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)    
        if 'text_encoder' in key:                
            if 'layer' in key:
                encoder_keys = key.split('.')
                layer_num = int(encoder_keys[4])
                if layer_num<6:
                    del state_dict[key]  
                    continue
                else:
                    decoder_layer_num = (layer_num-6)
                    encoder_keys[4] = str(decoder_layer_num)
                    encoder_key = '.'.join(encoder_keys)     
            else:
                encoder_key = key
            decoder_key = encoder_key.replace('text_encoder','text_decoder')  
            state_dict[decoder_key] = state_dict[key]     

            del state_dict[key]
    new_state_dict = {}
    for k, v in model.state_dict().items():
        new_state_dict[k] = state_dict.get('.'.join(k.split('.')[1:]), v)
    model.load_state_dict(new_state_dict, strict=False)
