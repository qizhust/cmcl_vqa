# Cross-Modal Contrastive Learning for Robust VQA

This repo is an implementation in PyTorch and support [METER](https://github.com/zdou0830/METER), [ViLT](https://github.com/dandelin/ViLT) and [ALBEF](https://github.com/salesforce/ALBEF) backbones.


## Data preparation and pretrained models

Please follow [METER](https://github.com/zdou0830/METER) and [ViLT](https://github.com/dandelin/ViLT/blob/master/DATA.md) to prepare the datasets and download the pretrained checkpoints released by corresponding backbones.


## Finetune on VQA data
### train with single-node multi-gpu
```bash
python -m torch.distributed.launch --nproc_per_node=1 main.py \
    --vision_encoder clip16 --kwargs \
    num_workers=4 \
    data_root=path/to/datasets/in/arrows/ \
    per_gpu_batchsize=8 \
    exp_name=finetune_vqa \
    resume_from=result/official_released/meter_clip16_288_roberta_pretrain.ckpt \
    load_pretrained_meter=1 \
    draw_false_image_vqa=1 \
    model_type=cmcl \
    cl_type=nce_graph \
    nce_lambda=0.5 \
    test_only=0 \
    dataset_name=vqacp \
    max_epochs=10
```

### train with multi-node multi-gpu
```bash
python main_dist.py --dist-url tcp://$server_ip:$port_id --world-size $num_machines --rank 0 --multiprocessing-distributed \
    --vision_encoder clip16 --kwargs \
    num_workers=4 \
    data_root=path/to/datasets/in/arrows/ \
    per_gpu_batchsize=8 \
    exp_name=finetune_vqa \
    resume_from=result/official_released/meter_clip16_288_roberta_pretrain.ckpt \
    load_pretrained_meter=1 \
    draw_false_image_vqa=1 \
    model_type=cmcl \
    cl_type=nce_graph \
    nce_lambda=0.5 \
    test_only=0 \
    dataset_name=vqacp \
    max_epochs=10
```
