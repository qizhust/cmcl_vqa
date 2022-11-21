from .cfg_basic import BasicConfig, _loss_names

task_mlm_itm_clip_bert = dict(
    exp_name = "mlm_itm",
    datasets = ["coco", "vg", "sbu", "gcc"],
    loss_names = _loss_names({"itm": 1, "mlm": 1}),
    batch_size = 4096,
    max_epochs = 10,
    max_steps = 100000,
    warmup_steps = 0.1,
    whole_word_masking = 1,

    vocab_size = 30522,
    max_text_len = 50,
    image_size = 224,
    tokenizer = "bert-base-uncased",
    train_transform_keys = ["clip"],
    val_transform_keys = ["clip"],
    learning_rate = 1e-5,
    val_check_interval = 1.0,
    lr_mult_head = 5,
    lr_mult_cross_modal = 5,
    num_top_layer = 6,
    hidden_size = 768,
    num_heads = 12,
)

task_finetune_vqa_clip_bert = dict(
    exp_name = "finetune_vqa",
    datasets = ["vqa"],
    loss_names = _loss_names({"vqa": 1}),
    batch_size = 512,
    max_epochs = 10,
    max_steps = -1,
    warmup_steps = 0.1,
    draw_false_image = 0,
    learning_rate = 5e-6,
    val_check_interval = 0.1,
    lr_mult_head = 50,
    lr_mult_cross_modal = 5,
    tokenizer = "bert-base-uncased",
    max_text_len = 50,
    input_text_embed_size = 768,
    vit = 'ViT-B/32',
    train_transform_keys = ["clip"],
    val_transform_keys = ["clip"],
    input_image_embed_size = 768,
    image_size = 576,
    dataset_name = "vqav2"
)

task_dict = {"task_mlm_itm_clip_bert": task_mlm_itm_clip_bert,
             "task_finetune_vqa_clip_bert": task_finetune_vqa_clip_bert}
