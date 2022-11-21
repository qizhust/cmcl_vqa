def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "vqa": 0,
    }
    ret.update(d)
    return ret

class BasicConfig():
    def __init__(self):
        self.exp_name = "multimodal"
        self.seed = 0
        self.datasets = ["coco", "vg", "sbu", "gcc"]
        self.loss_names = _loss_names({"itm": 1, "mlm": 1})
        self.batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
        self.train_split = 'train'  # or 'trainval'  whether using val data in training

        # Image setting
        self.train_transform_keys = ["clip"]
        self.val_transform_keys = ["clip"]
        self.image_size = 224
        self.patch_size = 32
        self.draw_false_image = 1
        self.image_only = 0  # replace boolean
        self.resolution_before = 224

        # Text Setting
        self.vqav2_label_size = 3129
        self.max_text_len = 40
        self.tokenizer = "bert-base-uncased"
        self.vocab_size = 30522
        self.input_text_embed_size = 768
        self.whole_word_masking = 0 # replace, boolean. note that whole_word_masking does not work for RoBERTa
        self.mlm_prob = 0.15
        self.draw_false_text = 0

        # Transformer Setting
        self.num_top_layer = 6
        self.input_image_embed_size = 768
        self.vit = 'ViT-B/32'
        self.hidden_size = 768
        self.num_heads = 12
        self.num_layers = 6
        self.mlp_ratio = 4
        self.drop_rate = 0.1

        # Optimizer Setting
        self.optim_type = "adamw"
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.decay_power = 1
        self.max_epochs = 100
        self.max_steps = 100000
        self.warmup_steps = 0.1
        self.end_lr = 0
        self.lr_mult_head = 5  # multiply lr for downstream heads
        self.lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

        # Downstream Setting
        self.get_recall_metric = 0  # replace boolean

        # PL Trainer Setting
        self.resume_from = ""
        self.fast_dev_run = 0  # replace boolean
        self.val_check_interval = 1.0
        self.test_only = 0  # replace boolean

        # below params varies with the environment
        self.data_root = ""
        self.log_dir = "result"
        self.per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
        self.load_path = ""
        self.num_workers = 8
        self.precision = 32
        self.load_pretrained_meter = 0  # replace boolean

        # contrastive learning
        self.cl_type = ""  # can be ["nce", "nce_graph", "nce_graph_multipos"]
        self.nce_lambda = 0.5
        self.answer_hidden_size = 128
        self.draw_false_image_vqa = 0  # currently only support 0, 1
        self.model_type = "meter"  # can be ["meter", "cmcl"]

        # whether freezing
        self.freeze_vision_encoder = 0

        # for vqacp
        self.dataset_name = ""

        # for albef
        self.distill = 0

    def update(self, dict_in):
        if dict_in is not None:
            for key, value in dict_in.items():
                setattr(self, key, type(getattr(self, key))(value))
