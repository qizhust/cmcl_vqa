import os
import argparse
from tqdm import tqdm
import glob

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from cfgs import create_config
from modules.optim_utils import set_schedule
from modules.pre_utils import save_hyperparameters, logging_scalars, ParseKwargs
from modules.wrap_utils import epoch_wrapup, vqa_test_step, vqa_test_wrapup
from models import build_model, load_ckpt
from dataset import build_dataloader, build_vqa_id2answer


def train_func(gpu, global_rank, world_size, task, vision_encoder, text_encoder, data_aug, kwargs):
    cfg = create_config(task, vision_encoder, text_encoder, data_aug, kwargs)
    ckpt_dir = f"{cfg.log_dir}/{cfg.exp_name}/{cfg.dataset_name}/{cfg.model_type}_{cfg.cl_type}_{cfg.nce_lambda}_{cfg.answer_hidden_size}_{cfg.vit.replace('/', '_')}_{cfg.tokenizer}_{cfg.image_size}_{cfg.train_split}_{world_size}_gpus_seed_{cfg.seed}"
    torch.cuda.manual_seed_all(cfg.seed)

    model = build_model(cfg, gpu)

    if cfg.test_only:
        ckpt_dir = cfg.load_path[:cfg.load_path.rfind('/')]
        _ = load_ckpt(model, cfg, cfg.load_path)
        dataloader, _, _ = build_dataloader(cfg, 'test')
        id2answer = build_vqa_id2answer(cfg)

        model.eval()
        outs = []
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            output = model(batch)
            if 'vqa' in model.module.current_tasks:
                outs.append(vqa_test_step(batch, output['non_scalars'], id2answer))
        if 'vqa' in model.module.current_tasks:
            vqa_test_wrapup(outs, ckpt_dir, global_rank)
        scalar_output = epoch_wrapup(model.module, model.module.current_tasks)

    else:
        if global_rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            writer = SummaryWriter(ckpt_dir)
            save_hyperparameters(cfg, ckpt_dir)
        else:
            writer = None
    
        dataloader = {}
        sampler = {}
        for split in [cfg.train_split, 'val']:
            dataloader[split], _, sampler[split] = build_dataloader(cfg, split)

        # define optimizer
        accum_iter = max(cfg.batch_size // (cfg.per_gpu_batchsize*world_size), 1)
        optimizer, scheduler = set_schedule(model, cfg.learning_rate, cfg.weight_decay, cfg.lr_mult_head, 
                                            cfg.lr_mult_cross_modal, cfg.end_lr, cfg.decay_power, 
                                            cfg.optim_type, cfg.warmup_steps, cfg.max_steps, 
                                            cfg.max_epochs, accum_iter, len(dataloader[cfg.train_split]))

        if cfg.resume_from != "":
            ckpt = load_ckpt(model, cfg, cfg.resume_from)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler['scheduler'].load_state_dict(ckpt['scheduler'])
                start_epoch = ckpt['epoch'] + 1
                best_metric = ckpt['best_metric']
                grad_step = ckpt['grad_step']
            else:
                start_epoch = 0
                best_metric = '0.'
                grad_step = 0
        else:
            start_epoch = 0
            best_metric = '0.'
            grad_step = 0

        # train-val
        for epoch in range(start_epoch, cfg.max_epochs):
            print(f"Epoch {epoch} starts!")
            # prepare for training
            model.train()
            sampler[cfg.train_split].set_epoch(epoch)
            dist.barrier()

            # start training
            for step, batch in tqdm(enumerate(dataloader[cfg.train_split], start=1), total=len(dataloader[cfg.train_split])):
                output = model(batch)
                total_loss = sum([v for k, v in output['scalars'].items() if "loss" in k])/accum_iter
                output['scalars'].update({'total_loss': total_loss})
                output['scalars']['total_loss'].backward()

                # update model
                if (step % accum_iter == 0) or (step == len(dataloader[cfg.train_split])):
                    grad_step += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler["scheduler"].step()
                    logging_scalars(global_rank, output['scalars'], 'train', writer, grad_step)

            scalar_output = epoch_wrapup(model.module, model.module.current_tasks)
            logging_scalars(global_rank, scalar_output, 'train', writer, grad_step)            

            # validation
            model.eval()
            with torch.no_grad():
                for val_batch in tqdm(dataloader['val'], total=len(dataloader['val'])):
                    output = model(val_batch)

            scalar_output = epoch_wrapup(model.module, model.module.current_tasks)
            logging_scalars(global_rank, scalar_output, 'val', writer, grad_step)
            if scalar_output['val/the_metric'] > float(best_metric):
                if global_rank == 0:
                    old_ckpt = glob.glob(os.path.join(ckpt_dir, f"*_metric_{best_metric}.pth"))
                    new_best = f"{scalar_output['val/the_metric'].item():.4f}"
                    new_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch}_metric_{new_best}.pth")
                    state = {'epoch': epoch, 'best_metric': new_best, 'grad_step': grad_step,
                             'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 
                             'scheduler': scheduler['scheduler'].state_dict()}
                    torch.save(state, new_ckpt)
                    
                    if len(old_ckpt) > 0:
                        os.remove(old_ckpt[0])
                    best_metric = new_best  # in string

            if epoch+1 == cfg.max_epochs:
                torch.save({'state_dict': model.state_dict()}, f"{ckpt_dir}/last.pth")


def spmd_main(local_rank, task, vision_encoder, text_encoder, data_aug, kwargs=None):
    # initialize the process group
    env_dict = {key: os.environ[key] for key in ["MASTER_ADDR", "MASTER_PORT", 
                "RANK", "WORLD_SIZE"]}
    print(f"[{os.getpid()}] Initialization process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
          + f"rank = {dist.get_rank()}, backend = {dist.get_backend()}")

    train_func(local_rank, local_rank, dist.get_world_size(), task, vision_encoder, text_encoder, data_aug, kwargs)
    dist.destroy_process_group()
    print("Exiting main function...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--task', type=str, default="task_finetune_vqa_clip_bert")
    parser.add_argument('--vision_encoder', type=str, default="clip32")
    parser.add_argument('--text_encoder', type=str, default="text_roberta")
    parser.add_argument('--data_aug', type=str, default="clip_randaug")
    parser.add_argument('--kwargs', nargs='*', action=ParseKwargs, help="other configure variables in kv pairs")
    args = parser.parse_args()

    spmd_main(args.local_rank, args.task, args.vision_encoder, args.text_encoder, args.data_aug, args.kwargs)
