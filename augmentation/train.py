import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch

import numpy as np
import re
from torch.distributed import get_rank, get_world_size

from model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Adam
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from gpt2_training.eval_utils import eval_model_loss

from data_loader import BucketingDataLoader, DynamicBatchingLoader
from gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list
from gpt2_training.train_utils import (InputFeatures, InputFeatures_train,RedditExample)
from gpt2_training.eval_utils import predict_sampling,predict_beam
END_OF_TEXT_TOKEN = '<|endoftext|>'
INF = 100000000
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_inputs_from_text(text, tokenizer,id_):
    end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
    src,tgt= text.strip().split('\t')
    src=re.sub(r'[^\w\s.!?"\',]','',src)
    tgt=re.sub(r'[^\w\s.!?"\',]','',tgt)
    lm_labels = []
    token_type_ids = [] 

    src_id = tokenizer.encode(src)
    tgt_id = tokenizer.encode(tgt)
    inputs=src_id+[end_of_text_id]
 
    lm_labels += [-1] * len(src_id)
    token_type_ids += [0] * len(src_id)
   
    inputs=inputs+tgt_id
    lm_labels += (tgt_id + [end_of_text_id])
    token_type_ids += [1] * (len(tgt_id) + 1)

    while len(inputs) % 8 != 0:
        inputs.append(0)
        token_type_ids.append(0)
        lm_labels.append(-1)

    position_ids = list(range(len(inputs)))
    assert (len(inputs) == len(position_ids) == len(token_type_ids)== len(lm_labels))
    assert len(inputs) % 8 == 0
    feature = InputFeatures_train(id_, inputs, position_ids, token_type_ids,lm_labels)
    return  feature

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)

parser.add_argument("--train_batch_size", type=int, default=4,help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--train_epoch", type=int, default=5)
parser.add_argument("--valid_step", type=int, default=500,help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=True)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

parser.add_argument('--local_rank', type=int, default=-1,help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')
args = parser.parse_args()

assert args.train_batch_size % args.gradient_accumulation_steps == 0, 'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size// args.gradient_accumulation_steps)

if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = os.path.join(args.output_dir,'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                               args.train_batch_size, n_gpu,
                                               timestamp))
log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
chunk = []
lens = []
n_example,sum=0,0
with open(args.train_input_file, "r", encoding="utf-8") as reader:
    for line in reader:
        feature = _get_inputs_from_text(line, enc,n_example)
        if feature.input_len > args.max_seq_length:
            continue
        chunk.append(feature)
        lens.append(feature.input_len)
        n_example+=1

config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
train_dataloader = BucketingDataLoader(chunk,lens,args.train_batch_size,args.max_seq_length)
eval_dataloader_loss = DynamicBatchingLoader(args.eval_input_file, enc, args.normalize_data,
    args.eval_batch_size, args.max_seq_length)

model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,args, verbose=True)
if args.local_rank != -1:
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(params, float(torch.distributed.get_world_size()))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']  
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,lr=args.learning_rate,bias_correction=False,max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,verbose=False)
    else:
        optimizer = FP16_Optimizer(optimizer,static_loss_scale=args.loss_scale,verbose=False)
else:
    optimizer = Adam(optimizer_grouped_parameters, args.learning_rate,max_grad_norm=1.0)

if args.local_rank == -1 or get_rank() == 0:
    train_logger = open(os.path.join(log_dir, 'train_log.txt'), 'a+', buffering=1)
    eval_logger = open(os.path.join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
    print('epoch,global_step,step,mean_loss,mean_ppl,n_token_real,n_token_total,epoch_time', file=train_logger)
    print('epoch,global_step,step,eval_loss,eval_ppl', file=eval_logger)

global_step,step,epoch = 0,0,0
num_optim_steps=args.train_epoch*n_example//args.train_batch_size
warmup_steps=args.warmup_proportion*num_optim_steps

if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=num_optim_steps, desc=f"training")
    else:
        pbar = None

for i in range(args.train_epoch):
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        seq_len = batch[0].shape[1]
        batch = tuple(t.to(device) for t in batch)
        input_ids, position_ids, token_ids, label_ids, *_ = batch
        if args.no_token_id:
            token_ids = None
        loss, ppl = model(input_ids, position_ids, token_ids, label_ids)

        if n_gpu > 1:
            loss = loss.mean()
            ppl = ppl.mean()
        loss = loss / (args.train_batch_size / input_ids.shape[0])
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        tr_loss += float(loss.item()) * (args.train_batch_size / input_ids.shape[0])
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps
        if ppl.item() < INF:
            tr_ppl += ppl.item()
        else:
            tr_ppl += mean_ppl
        mean_ppl = tr_ppl / nb_tr_steps

        n_token_total += input_ids.shape[0] * input_ids.shape[1]
        n_token_real += (input_ids != 0).sum().item()

        step += 1
        if step % args.gradient_accumulation_steps == 0:
            set_lr(optimizer, global_step,args.lr_schedule, args.learning_rate,warmup_steps, args.warmup_proportion,
                   config.n_embd, num_optim_steps)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total

            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"tok/s: {n_token_real_all_proc//epoch_time//1000}k "
                        f"ppl: {mean_ppl:.2f} epoch: {epoch}")
                    pbar.update(1)
                print('{},{},{},{},{},{},{},{}'.format(
                    epoch+1, global_step+1, step+1, mean_loss, mean_ppl,
                    n_token_real_all_proc, n_token_total_all_proc, epoch_time),file=train_logger)

            if global_step % args.valid_step == 0:
                if args.local_rank == -1 or get_rank() == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    eval_loss, eval_ppl = eval_model_loss(model, enc, eval_dataloader_loss, epoch, args)
                    print('{},{},{},{},{}'.format(epoch+1, global_step+1, step+1, eval_loss, eval_ppl),file=eval_logger)
                    logger.info('current learning rate: '+ str(optimizer.param_groups[0]['lr']))
                    model.train()


if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    train_logger.close()
    eval_logger.close()