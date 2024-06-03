import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
from os.path import join as pjoin
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import set_seed
import torch
from transformers import LlamaForCausalLM
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

from args import get_args
from dataset import (AVSDDataSet, collate_batch, get_dataset)
from utils.early_stop import EarlyStopping
import utils.mizhi as mizhi
import utils.lr_scheduler as lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

# from utils.tokenizer import Tokenizer
# tokenizer = Tokenizer(model_path='statics/download/llama-2-models_hf/7B/tokenizer.model')

def seed_worker(worker_id):
    # https://mp.weixin.qq.com/s/NgPChyIr2uLhONJHFS_iCw
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def get_two_loader(args):
    train_data = get_dataset(data_split='train',
                             args=args)
    train_dataset = AVSDDataSet(train_data, args)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_per_gpu,
                              shuffle=True,
                              collate_fn=lambda x: collate_batch(x, args),
                              pin_memory=False,
                              num_workers=4,
                              worker_init_fn=seed_worker)
    valid_data = get_dataset(data_split='dev',
                            args=args)
    valid_dataset = AVSDDataSet(valid_data, args)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.val_per_gpu,
                              shuffle=False,
                              collate_fn=lambda x: collate_batch(x, args),
                              pin_memory=False,
                              num_workers=4,
                              worker_init_fn=seed_worker)
    return train_loader, valid_loader

def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def do_train(args):
    ##############
    print("Setup Data")
    train_loader, valid_loader = get_two_loader(args)
    ##############
    print('Setup Train Model')
    model_path = 'ssg_llama2/statics/debug.llama2_7b'
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    ##############
    print('Setup Optimizer & Scaler')
    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    ##############
    early_stop = EarlyStopping(patience=100)
    min_val_loss = float('inf')
    for epoch in range(0, args.n_epochs):
        print(f'Epoch {epoch} Train Start')
        start_t = time.time()
        train_loss_sum = 0
        # ------------- train ----------------
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc='Train', mininterval=10)):
            # we use a per iteration (instead of per epoch) lr scheduler
            if step % args.gradient_accumulation_steps == 0:
                lr_sched.adjust_learning_rate(optimizer, step / len(train_loader) + epoch, args)
            # TBoard.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + step)
            batch = [value.to('cuda:0', non_blocking=True) if isinstance(value, torch.Tensor) else value for index, value in enumerate(batch)]
            input_ids, lm_labels_list, input_mask = batch
            
            with autocast(dtype=torch.bfloat16):
                train_loss = model(
                    input_ids=input_ids,
                    labels=lm_labels_list,
                    attention_mask=input_mask,
                )[0]

            if not math.isfinite(train_loss.item()):
                print("Train Loss is {}, stopping training".format(train_loss.item()))
                sys.exit(1)
            train_loss_sum += train_loss.item()
            
            train_loss = train_loss / args.gradient_accumulation_steps
            loss_scaler(
                loss=train_loss,
                optimizer=optimizer,
                clip_grad=args.max_norm,
                parameters=model.parameters(),
                update_grad=(step + 1) % args.gradient_accumulation_steps == 0
                )
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

        train_loss_avg = train_loss_sum / len(train_loader)      
        # ------------- valid ----------------
        print(f'Val Start')
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            for step, batch in enumerate(tqdm(valid_loader, desc='Val', mininterval=10)):
                batch = [value.to('cuda:0', non_blocking=True) if isinstance(value, torch.Tensor) else value for index, value in enumerate(batch)]
                input_ids, lm_labels_list, input_mask = batch
                
                with autocast(dtype=torch.bfloat16):
                    val_loss = model(
                        input_ids=input_ids,
                        labels=lm_labels_list,
                        attention_mask=input_mask
                    )[0]
                val_loss_sum += val_loss.item()
        val_loss_avg = val_loss_sum / len(valid_loader)
        # ------------- Save  ----------------
        print('Saving ...')
        TBoard_data = {
            'epoch': epoch,
            'Loss/valid': val_loss_avg,
            'Loss/train': train_loss_avg,
        }
        for name, value in TBoard_data.items():
            if name == 'epoch': 
                continue
            # TBoard.add_scalar(name, value, epoch)
        with open(pjoin('ssg_llama2/alog', args.exp, 'state.log'), 'a') as f:
            f.write(json.dumps({**TBoard_data, **(args.__dict__)}) + '\n')
        if val_loss_avg < min_val_loss:
            best_model_path = pjoin('ssg_llama2/alog', args.exp, 'best_model')
            os.popen(f"rm -rf {best_model_path}").read()
            model.save_pretrained(best_model_path)
            print(f'Epoch {epoch} best, saved in {best_model_path}! ({val_loss_avg} < {min_val_loss})')
            min_val_loss = val_loss_avg

        print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
        print('This epoch consume: {:.2f} min'.format((time.time() - start_t)/60))
        early_stop.record(val_loss_avg)
        if early_stop.early_stop:
            break
    print('*'*8 + ' Finish ' + '*'*8)
    os.popen(f'echo > {pjoin("ssg_llama2/alog", args.exp, ".is_done")}').read()


if __name__ == "__main__":
    common_args, args = get_args('train')
    
    assert common_args.task is not None
    if os.path.exists(f'ssg_llama2/alog/{args.exp}'):
        if args.exp not in ('try','try1','try2','try3'):
            assert input(f'Are you sure to delete "{args.exp}"? ').strip() in ('Y', 'y')
        print('Deleting', f'ssg_llama2/alog/{args.exp} ...')
        shutil.rmtree(f'ssg_llama2/alog/{args.exp}')

    os.makedirs(pjoin('ssg_llama2/alog', args.exp), exist_ok=True)
    with open(pjoin('ssg_llama2/alog', common_args.exp, 'common_args.json'), 'w') as f:
        json.dump(common_args.__dict__, f, indent=4)

    log_mode = 'w'
    sys.stdout = mizhi.Printer(pjoin('ssg_llama2/alog', args.exp, 'train.log'), log_mode)
    sys.stderr = mizhi.Errorer(pjoin('ssg_llama2/alog', args.exp, 'train.error'), log_mode)
    print('Start at: ' + time.strftime("%m-%d %H:%M:%S", time.localtime()))
    print('Run Place:', socket.gethostbyname(socket.gethostname())+f'({torch.cuda.get_device_name()} cuda:{os.environ["CUDA_VISIBLE_DEVICES"]})',
                         'Conda env:', os.environ['CONDA_DEFAULT_ENV'])
    pprint(vars(args))

    set_seed(args.seed)

    TBoard = SummaryWriter(f'ssg_llama2/alog/{args.exp}/tb')
    try:
        do_train(args)
    except BaseException as e:
        traceback.print_exc()
        TBoard.close()
    