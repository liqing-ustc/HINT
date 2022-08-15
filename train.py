import time
from tqdm import tqdm, trange
from collections import Counter, OrderedDict

from dataset import HINT, HINT_collate
from model import make_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import argparse
import sys
import os
from torch.optim import Adam
from optimization import AdamW, WarmupLinearSchedule, ConstantLRSchedule
from utils import *
from result_encoding import ResultEncoding

def parse_args():
    parser = argparse.ArgumentParser('Give Me A HINT')
    parser.add_argument('--wandb', type=str, default='HINT', help='the project name for wandb.')
    parser.add_argument('--resume', type=str, default=None, help='Resumes training from checkpoint.')
    parser.add_argument('--perception_pretrain', type=str, help='initialize the perception from pretrained models.',
                        default='data/perception_pretrain/model.pth.tar_78.2_match')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory for storing checkpoints')
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")

    parser.add_argument('--model', type=str, default='LSTM', 
            choices=['LSTM', 'LSTM_attn', 'GRU', 'GRU_attn', 'ON', 'OM', 'TRAN.opennmt', 'TRAN.relative', 'TRAN.relative_universal'],
            help='the type of model: GRU, LSTM, TRAN for transformer, ON for Ordered Neuron LSTM, OM for Ordered Memory.')
    parser.add_argument('--nhead', type=int, default=1, help="number of attention heads in the Transformer model")
    parser.add_argument('--layers', type=int, default=1, help="number of layers for both encoder and decoder")
    parser.add_argument('--enc_layers', type=int, default=0, help="number of layers in encoder")
    parser.add_argument('--dec_layers', type=int, default=0, help="number of layers in decoder")
    parser.add_argument('--emb_dim', type=int, default=128, help="embedding dim")
    parser.add_argument('--hid_dim', type=int, default=128, help="hidden dim")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout ratio")

    parser.add_argument('--train_size', type=float, default=None, help="what perceptage of train data is used.")
    parser.add_argument('--max_op_train', type=int, default=None, help="The maximum number of ops in train.")
    parser.add_argument('--main_dataset_ratio', type=float, default=0, 
            help="The percentage of data from the main training set to avoid forgetting in few-shot learning.")
    parser.add_argument('--fewshot', default=None, choices=list('xyabcd'), help='fewshot concept.')
    parser.add_argument('--input', default='image', choices=['image', 'symbol'], help='whether to provide perfect perception, i.e., no need to learn')
    parser.add_argument('--curriculum', default='no', choices=['no', 'manual'], help='whether to use the pre-defined curriculum')
    parser.add_argument('--pos_emb_type', default='sin', choices=['sin', 'learn'])
    parser.add_argument('--save_model', default='False', choices=['True', 'False'])
    parser.add_argument('--result_encoding', default='decimal', choices=['decimal', 'binary', 'sin'])
    parser.add_argument('--cos_sim_margin', type=float, default=0.2, 
                    help='the margin used to compute the loss for sin result encoding.')
    parser.add_argument('--max_rel_pos', type=int, default=15, help='the maximum relative position used in relative transformer.')
    parser.add_argument('--output_attentions', action='store_true', help='output attentions for visualization of Transformer.')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_scheduler', default='constant', choices=['constant', 'warmup'])
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--iterations', type=int, default=None, help='number of iterations for training')
    parser.add_argument('--iterations_eval', type=int, default=None, help='how many iterations per evaluation')
    parser.add_argument('--early_stop', type=int, default=None, help='stop training if the model does not improve for x evaluations.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--epochs_eval', type=int, default=1, help='how many epochs per evaluation')
    args = parser.parse_args()
    args.enc_layers = args.enc_layers or args.layers
    args.dec_layers = args.dec_layers or args.layers
    args.save_model = args.save_model == 'True'
    return args

def evaluate(model, dataloader, args, log_prefix='val'):
    model.eval() 
    res_all = []
    res_pred_all = []

    expr_all = []
    expr_pred_all = []

    dep_all = []
    dep_pred_all = []

    metrics = OrderedDict()

    with torch.no_grad():
        for sample in tqdm(dataloader):
            if args.input == 'image':
                src = sample['img_seq']
            elif args.input == 'symbol':
                src = torch.tensor([x for s in sample['sentence'] for x in s])
            res = sample['res']
            if args.result_encoding == 'sin':
                tgt = res.unsqueeze(1)
            else:
                tgt = torch.tensor(args.res_enc.res2seq_batch(res.numpy()))
            expr = sample['expr']
            dep = sample['head']
            src_len = sample['len']

            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            output = model(src, tgt[:, :-1], src_len)
            pred = torch.argmax(output, -1).detach().cpu().numpy()
            if args.result_encoding == 'sin':
                res_pred = pred
            else:
                res_pred = args.res_enc.seq2res_batch(pred)
            res_pred_all.append(res_pred)
            res_all.append(res)

            # expr_pred_all.extend(expr_preds)
            expr_all.extend(expr)
            # dep_pred_all.extend(dep_preds)
            dep_all.extend(dep)

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    result_acc = (res_pred_all == res_all).mean()
    metrics['result_acc/avg'] = result_acc

    tracked_attrs = ['length', 'symbol', 'digit', 'result', 'eval', 'tree_depth', 'ps_depth']
    for attr in tracked_attrs:
        # print(f"result accuracy by {attr}:")
        attr2ids = getattr(dataloader.dataset, f'{attr}2ids')
        for k, ids in sorted(attr2ids.items()):
            res = res_all[ids]
            res_pred = res_pred_all[ids]
            res_acc = (res == res_pred).mean() if ids else 0.
            k = 'div' if k == '/' else k
            metrics[f'result_acc/{attr}/{k}'] = res_acc
            # print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    wandb.log({f'{log_prefix}/{k}': v for k, v in metrics.items()})
    
    # print("error cases:")
    # errors = np.arange(len(res_all))[res_all != res_pred_all]
    # for i in errors[:10]:
    #     print(expr_all[i], dep_all[i], res_all[i], res_pred_all[i])

    return 0., 0., result_acc

def train(model, args, st_iter=0):
    best_acc = 0.0
    stop_counter = 0
    batch_size = args.batch_size
    train_dataloader = torch.utils.data.DataLoader(args.train_set, batch_size=batch_size,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)
    eval_dataloader = torch.utils.data.DataLoader(args.val_set, batch_size=32,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)

    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.lr_scheduler == 'constant':
        lr_scheduler = ConstantLRSchedule(optimizer)
    elif args.lr_scheduler == 'warmup':
        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs*len(train_dataloader),
                     last_epoch=st_iter-1)
    
    if args.result_encoding == 'sin':
        # criterion = nn.MultiMarginLoss(margin=args.cos_sim_margin)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.res_enc.null_idx)
    
    ##########evaluate init model###########
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader, args)
    print('Iter {}: {} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format(0, 'val', 100*perception_acc, 100*head_acc, 100*result_acc))
    ########################################

    train_iter = iter(train_dataloader)
    model.train()
    for step in trange(st_iter, args.iterations):
        try:
            sample = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            sample = next(train_iter)

        if args.input == 'image':
            src = sample['img_seq']
        elif args.input == 'symbol':
            src = torch.tensor([x for s in sample['sentence'] for x in s])
        res = sample['res']
        if args.result_encoding == 'sin':
            tgt = res.unsqueeze(1)
        else:
            tgt = torch.tensor(args.res_enc.res2seq_batch(res.numpy()))
        src_len = sample['len']

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        output = model(src, tgt[:, :-1], src_len)
        if args.result_encoding == 'sin':
            loss = criterion(output, tgt.flatten())
            # loss = -output.gather(1, tgt).mean()
        else:
            loss = criterion(output.contiguous().view(-1, output.shape[-1]), tgt[:, 1:].contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        lr_scheduler.step()

        pred = torch.argmax(output, -1)
        if args.result_encoding == 'sin':
            acc = pred == tgt.flatten()
        else:
            acc = torch.logical_or(pred == tgt[:, 1:], tgt[:, 1:] == args.res_enc.null_idx)
            acc = acc.all(axis=1)
        acc = acc.float().mean()

        wandb.log({'train/step': step, 'train/loss': loss.cpu().item(), 
                'train/result_acc': acc.cpu().item(), 'train/lr': lr_scheduler.get_last_lr()[0]})
            
        if ((step+1) % args.iterations_eval == 0) or (step+1 == args.iterations):
            perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader, args)
            print('Iter {}: {} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format(step+1, 'val', 100*perception_acc, 100*head_acc, 100*result_acc))
            model.train()

            if result_acc > best_acc:
                best_acc = result_acc
                stop_counter = 0
            else:
                stop_counter += 1
                if args.early_stop and stop_counter == args.early_stop:
                    print(f'Stop training because model does not improve for {stop_counter} evaluations.')
                    break

    wandb.log({'train_steps': step+1})
    if args.save_model:
        model_path = os.path.join(args.ckpt_dir, f'model_{step+1}.p')
        torch.save({'step': step+1, 'model_state_dict': model.state_dict()}, model_path)
        print(f'Save model to {model_path}.')
    # Test
    print('-' * 30)
    print('Evaluate on test set...')
    eval_dataloader = torch.utils.data.DataLoader(args.test_set, batch_size=64,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader, args, log_prefix='test')
    print('Iter {}: {} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format(args.iterations, 'test', 100*perception_acc, 100*head_acc, 100*result_acc))
    return



if __name__ == "__main__":
    args = parse_args()
    sys.argv = sys.argv[:1]
    wandb.init(project=args.wandb, dir=args.output_dir, config=vars(args))
    ckpt_dir = os.path.join(wandb.run.dir, '../ckpt')
    os.makedirs(ckpt_dir)
    args.ckpt_dir = ckpt_dir
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)

    # train_set = HINT('train', numSamples=5000)
    train_set = HINT('train', input=args.input, fewshot=args.fewshot, 
                    n_sample=args.train_size, max_op=args.max_op_train,
                    main_dataset_ratio=args.main_dataset_ratio)
    val_set = HINT('val', input=args.input, fewshot=args.fewshot)
    test_set = HINT('test', input=args.input, fewshot=args.fewshot)
    print('train:', len(train_set), 'val:', len(val_set), 'test:', len(test_set))
    
    args.res_enc = ResultEncoding(args.result_encoding)

    model = make_model(args)
    if args.resume:
        print('Load checkpoint from ' + args.resume)
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)

    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    wandb.log({'n_params': n_params})
    wandb.log({'train_examples': len(train_set)})
    print('Num params:', n_params)
    args.train_set = train_set
    args.val_set = val_set
    args.test_set = test_set

    if not args.iterations:
        args.iterations = args.epochs * len(train_set)
    if not args.iterations_eval:
        args.iterations_eval = args.epochs_eval * len(train_set)

    train(model, args)

