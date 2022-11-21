swin32_base224 = dict(
    vit = "swin_base_patch4_window7_224_in22k",
    patch_size = 32,
    image_size = 224,
    train_transform_keys = ["imagenet"],
    val_transform_keys = ["imagenet"],
    input_image_embed_size = 1024,
    resolution_before = 224,
)

swin32_base384 = dict(
    vit = "swin_base_patch4_window12_384_in22k",
    patch_size = 32,
    image_size = 384,
    train_transform_keys = ["imagenet"],
    val_transform_keys = ["imagenet"],
    input_image_embed_size = 1024,
    resolution_before = 384,
)

swin32_large384 = dict(
    vit = "swin_large_patch4_window12_384_in22k",
    patch_size = 32,
    image_size = 384,
    train_transform_keys = ["imagenet"],
    val_transform_keys = ["imagenet"],
    input_image_embed_size = 1536,
    resolution_before = 384,
)

clip32 = dict(
    vit = 'ViT-B/32',
    image_size = 224,
    patch_size = 32,
    train_transform_keys = ["clip"],
    val_transform_keys = ["clip"],
    input_image_embed_size = 768,
)

clip16 = dict(
    vit = 'ViT-B/16',
    image_size = 224,
    patch_size = 16,
    train_transform_keys = ["clip"],
    val_transform_keys = ["clip"],
    input_image_embed_size = 768,
)

vilt = dict(
    vit = "vit_base_patch32_384",
    image_size = 384,
    patch_size = 32,
    hidden_size = 768,
    num_heads = 12,
    num_layers = 12,
    train_transform_keys = ["pixelbert"],
    val_transform_keys = ["pixelbert"],
)

vision_dict = {"swin32_base224": swin32_base224,
               "swin32_base384": swin32_base384,
               "swin32_large384": swin32_large384,
               "clip32": clip32,
               "clip16": clip16,
               "vilt": vilt}
