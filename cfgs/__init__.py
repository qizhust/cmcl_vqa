from .cfg_basic import BasicConfig
from .cfg_task import task_dict
from .cfg_vision import vision_dict
from .cfg_text import text_dict
from .cfg_dataaug import aug_dict

def create_config(task=None, vision_encoder=None, text_encoder=None, data_aug=None, kwargs=None):

    configure = BasicConfig()
    if task is not None:
        configure.update(task_dict[task])
    if vision_encoder is not None:
        configure.update(vision_dict[vision_encoder])
    if text_encoder is not None:
        configure.update(text_dict[text_encoder])
    if data_aug is not None:
        configure.update(aug_dict[data_aug])
    if kwargs is not None:
        configure.update(kwargs)

    return configure
