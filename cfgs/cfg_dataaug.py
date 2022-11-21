imagenet_randaug = dict(
    train_transform_keys = ["imagenet_randaug"],
)

clip_randaug = dict(
    train_transform_keys = ["clip_randaug"],
)

pixelbert_randaug = dict(
    train_transform_keys = ["pixelbert_randaug"],
)

aug_dict = {"imagenet_randaug": imagenet_randaug,
            "clip_randaug": clip_randaug,
            "pixelbert_randaug": pixelbert_randaug}
