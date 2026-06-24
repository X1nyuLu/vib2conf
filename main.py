#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Description :
@InitTime    :2025/11/18 19:48:45
@Author      :XinyuLu
@EMail       :xinyulu@stu.xmu.edu.cn

'''



import uuid
import os
import time
import torch
import logging
import argparse
import yaml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.engine import seed_everything

import models
import trainers
import warnings


warnings.simplefilter("ignore", FutureWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('Vib2Conf', add_help=True)

    # ==================== Basic Parameters ====================
    parser.add_argument('--model', default='vibraclip',
                        help="Name of the network architecture")
    parser.add_argument('--launch', default='base',
                        help="Training launch config (selects loss combination and training strategy)")
    parser.add_argument('--ds', default='nist',
                        help="Dataset name")
    parser.add_argument('--task', default='ir',
                        help='Prediction task on the selected dataset')
    parser.add_argument('--grad-props', default=None,
                        help="Properties to be predicted via gradient computation")
    
    # ==================== Run Mode ====================
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true',
                            help="start train")
    mode_group.add_argument('--debug', action='store_true',
                            help="start debug")

    # ==================== Device & Distributed Training ====================
    parser.add_argument('--device', default='cuda:0',
                        help="GPU device to run on")
    parser.add_argument('--base-model-path',
                        default='',
                        help="Path to the pretrained base model (for fine-tuning)")
    parser.add_argument('--seed', default=624, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument('--ddp', '-ddp', action='store_true', default=False,
                        help="Enable DistributedDataParallel (DDP) training")
    parser.add_argument('--force-reload', action='store_true', default=False,
                        help="Force reload datasets from scratch")

    # ==================== Training Strategy ====================
    parser.add_argument('--batch-size', type=int,
                        help="Batch size for training")
    parser.add_argument('--epoch', type=int,
                        help="Total number of training epochs")
    parser.add_argument('--lr', type=float,
                        help="Initial learning rate")
    parser.add_argument('--config', default=None, type=str,
                        help="Path to the hyper-parameter config file")
    parser.add_argument('--grad-norm', type=float, default=1.0,
                        help="Maximum norm for gradient clipping")

    # ==================== Checkpoint ====================
    parser.add_argument('--find-unused-parameters', action='store_true', default=False,
                        help="Set find_unused_parameters=True in DDP (useful when some params receive no gradient)")
    parser.add_argument('--use-ema', action='store_true', dest='use_ema', default=False,
                        help="Enable Exponential Moving Average (EMA)")

    args = parser.parse_args()
    return args


def init_logs(args, local_rank, ts, random_id):
    if args.debug:
        return

    os.makedirs(f'logs/{args.ds}/{args.task}/{args.model}', exist_ok=True)

    if local_rank == 0:
        logging.basicConfig(
            filename=f'logs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}.log',
            format='%(levelname)s:%(message)s',
            level=logging.INFO
            )

        logging.info({k: v for k, v in args.__dict__.items() if v})
        print(f'logging save path: ./logs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}.log')


def init_device(args):
    if args.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        return torch.device("cuda", local_rank)
    return args.device


def init_model(args, device, local_rank, ts, random_id):

    if args.train and not args.debug:
        if local_rank == 0:
            os.makedirs(f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}", exist_ok=True)

    with open('config.yaml', "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.pop('defaults')
    task_config = config[args.launch] if args.config is None else config[args.config]
    params = defaults.copy()
    params.update(task_config)

    if args.batch_size:
        params['batch_size'] = args.batch_size
    if args.epoch:
        params['epoch'] = args.epoch
    if args.lr:
        params["lr"] = args.lr
    if args.debug:
        params['epoch'] = 1

    model = models.build_model(args.model, **params)

    if 'cuda' in args.device and not args.ddp:
        model = model.to(device)

    base_model_path = args.base_model_path
    if base_model_path:
        ckpt = torch.load(base_model_path, map_location='cpu', weights_only=True)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)

    if args.ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        ddp_device = torch.device("cuda", local_rank)

        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method('spawn')
        model = model.to(ddp_device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.find_unused_parameters)

    return model, params


if __name__ == "__main__":

    args = get_args_parser()
    device = init_device(args)
    local_rank = 0 if not args.ddp else int(os.environ["LOCAL_RANK"])

    seed_everything(args.seed)

    if args.grad_props:
        task_set = set(args.task.split('-'))
        for gp in args.grad_props.split('-'):
            assert gp in task_set, (
                f"grad-props element '{gp}' must also be present in --task '{args.task}'. "
                f"Available tasks: {sorted(task_set)}"
            )

    ts = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    random_id = uuid.uuid4().hex[:6]

    model_save_path = f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}"

    if args.debug:
        model_save_path = None

    model, params = init_model(args, device, local_rank, ts, random_id)
    init_logs(args, local_rank, ts, random_id)
    logging.info({k: v for k, v in params.items()})

    if args.train or args.debug:
        trainers.launch_training(
            args.launch,
            model=model,
            ds=args.ds,
            task=args.task,
            data_dir='datasets',
            model_save_path=model_save_path,
            device=device,
            ddp=args.ddp,
            rank=local_rank,
            config=params,
            force_reload=args.force_reload,
            grad_norm=args.grad_norm,
            use_ema=args.use_ema,
        )