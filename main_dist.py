import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modules.pre_utils import ParseKwargs
from main import train_func

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.multiprocessing_distributed:
        print('Multiprocessing_distributed enabled!')
        args.rank = args.rank * ngpus_per_node + gpu

    # initialize the process group
    print(f"Start dist initialization with {args.dist_url}, world_size={args.world_size}, global_rank={args.rank}, gpu_id={gpu}")
    dist.init_process_group(backend="nccl", init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    print("Finish dist initialization")

    train_func(gpu, args.rank, args.world_size, args.task, args.vision_encoder, args.text_encoder, args.data_aug, args.kwargs)
    dist.destroy_process_group()
    print("Exiting main function...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=-1, 
                        help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--task', type=str, default="task_finetune_vqa_clip_bert")
    parser.add_argument('--vision_encoder', type=str, default="clip32")
    parser.add_argument('--text_encoder', type=str, default="text_roberta")
    parser.add_argument('--data_aug', type=str, default="clip_randaug")
    parser.add_argument('--kwargs', nargs='*', action=ParseKwargs, help="other configure variables in kv pairs")
    args = parser.parse_args()

    print('Entering main!')
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
