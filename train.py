from baseline_utils import *
import time
from tqdm import tqdm
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from dataset import HINT, HINT_collate
from model import make_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import sys
from torch.optim import Adam
from optimization import AdamW, WarmupLinearSchedule, ConstantLRSchedule
from baseline_utils import SYMBOLS, INP_VOCAB, RES_VOCAB, DEVICE, NULL, END, RES_MAX_LEN

def parse_args():
    parser = argparse.ArgumentParser('Give Me A HINT')
    parser.add_argument('--resume', type=str, default=None, help='Resumes training from checkpoint.')
    parser.add_argument('--perception-pretrain', type=str, help='initialize the perception from pretrained models.',
                        default='data/perception-pretrain/model.pth.tar_78.2_match')
    parser.add_argument('--output-dir', type=str, default='outputs/', help='output directory for storing checkpoints')
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")

    parser.add_argument('--seq2seq', type=str, default='GRU', help='the type of seq2seq: GRU, LSTM, TRAN for Transformer, ON for Ordered Neuron LSTM, OM for Ordered Memory.')
    parser.add_argument('--nhead', type=int, default=1, help="number of attention heads in the Transformer model")
    parser.add_argument('--enc_layers', type=int, default=1, help="number of layers in encoder")
    parser.add_argument('--dec_layers', type=int, default=1, help="number of layers in decoder")
    parser.add_argument('--emb_dim', type=int, default=128, help="embedding dim")
    parser.add_argument('--hid_dim', type=int, default=128, help="hidden dim")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout ratio")

    parser.add_argument('--perception', action='store_true', help='whether to provide perfect perception, i.e., no need to learn')
    parser.add_argument('--syntax', action='store_true', help='whether to provide perfect syntax, i.e., no need to learn')
    parser.add_argument('--semantics', action='store_true', help='whether to provide perfect semantics, i.e., no need to learn')
    parser.add_argument('--curriculum', action='store_true', help='whether to use the pre-defined curriculum')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--epochs_eval', type=int, default=10, help='how many epochs per evaluation')
    args = parser.parse_args()
    return args

from nltk.tree import Tree
def draw_parse(sentence, head):
    def build_tree(pos):
        children = [i for i, h in enumerate(head) if h == pos]
        return Tree(sentence[pos], [build_tree(x) for x in children])
    
    root = head.index(-1)
    tree = build_tree(root)
    return tree

def evaluate(model, dataloader):
    model.eval() 
    res_all = []
    res_pred_all = []

    expr_all = []
    expr_pred_all = []

    dep_all = []
    dep_pred_all = []

    with torch.no_grad():
        for sample in tqdm(dataloader):
            img = sample['img_seq']
            src = torch.tensor([x for s in sample['sentence'] for x in s])
            res = sample['res']
            tgt = torch.tensor(res2seq(res.numpy()))
            expr = sample['expr']
            dep = sample['head']
            src_len = sample['len']
            tgt_len = [len(str(x)) for x in res.numpy()]

            img = img.to(DEVICE)
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            output = model(img, src, tgt[:, :-1], src_len, tgt_len)
            pred = torch.argmax(output, -1).detach().cpu().numpy()
            res_pred = [seq2res(x) for x in pred]
            res_pred_all.append(res_pred)
            res_all.append(res)

            # expr_pred_all.extend(expr_preds)
            expr_all.extend(expr)
            # dep_pred_all.extend(dep_preds)
            dep_all.extend(dep)

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    result_acc = (res_pred_all == res_all).mean()

    print("result accuracy by length:")
    for k in sorted(dataloader.dataset.len2ids.keys()):
        ids = dataloader.dataset.len2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))
    
    print("result accuracy by symbol:")
    for k in sorted(dataloader.dataset.sym2ids.keys()):
        ids = dataloader.dataset.sym2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    print("result accuracy by digit:")
    for k in sorted(dataloader.dataset.digit2ids.keys()):
        ids = dataloader.dataset.digit2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    print("result accuracy by result:")
    for k in sorted(dataloader.dataset.res2ids.keys())[:10]:
        ids = dataloader.dataset.res2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    print("result accuracy by generalization:")
    for k in sorted(dataloader.dataset.cond2ids.keys()):
        ids = dataloader.dataset.cond2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        if len(ids) == 0:
            res_acc = 0.
        else:
            res_acc = (res == res_pred).mean()
        print(k, "(%.2f%%)"%(100*len(ids)/len(dataloader.dataset)), "%5.2f"%(100 * res_acc))
    
    print("error cases:")
    errors = np.arange(len(res_all))[res_all != res_pred_all]
    for i in errors[:10]:
        print(expr_all[i], dep_all[i], res_all[i], res_pred_all[i])

    return 0., 0., result_acc

def train(model, args, st_epoch=0):
    best_acc = 0.0
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(args.train_set, batch_size=batch_size,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)
    eval_dataloader = torch.utils.data.DataLoader(args.val_set, batch_size=32,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)

    # optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    # lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10 * len(train_dataloader), t_total=args.epochs*len(train_dataloader),
    #                  last_epoch=st_epoch*len(train_dataloader)-1)
    optimizer = Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ConstantLRSchedule(optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=RES_VOCAB.index(NULL))
    
    max_len = float("inf")
    if args.curriculum:
        curriculum_strategy = dict([
            # (0, 7)
            (0, 1),
            (1, 3),
            (20, 7),
            (40, 11),
            (60, 15),
            (80, float('inf')),
        ])
        print("Curriculum:", sorted(curriculum_strategy.items()))
        for e, l in sorted(curriculum_strategy.items(), reverse=True):
            if st_epoch >= e:
                max_len = l
                break
        train_set.filter_by_len(max_len=max_len)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=4, collate_fn=HINT_collate)
    
    ##########evaluate init model###########
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader)
    print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('val', 100*perception_acc, 100*head_acc, 100*result_acc))
    ########################################

    for epoch in range(st_epoch, args.epochs):
        if args.curriculum and epoch in curriculum_strategy:
            max_len = curriculum_strategy[epoch]
            train_set.filter_by_len(max_len=max_len)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=4, collate_fn=HINT_collate)

        since = time.time()
        print('-' * 30)
        print('Epoch {}/{} (max_len={}, data={}, lr={})'.format(epoch, args.epochs - 1, max_len, len(train_set), lr_scheduler.get_lr()[0]))

        model.train()
        train_acc = []
        train_loss = []
        for sample in tqdm(train_dataloader):
            img = sample['img_seq']
            src = torch.tensor([x for s in sample['sentence'] for x in s])
            res = sample['res']
            tgt = torch.tensor(res2seq(res.numpy()))
            src_len = sample['len']
            tgt_len = [len(str(x)) for x in res.numpy()]

            img = img.to(DEVICE)
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            output = model(img, src, tgt[:, :-1], src_len, tgt_len)
            loss = criterion(output.contiguous().view(-1, output.shape[-1]), tgt[:, 1:].contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            lr_scheduler.step()
            train_loss.append(loss.cpu().item())

            pred = torch.argmax(output, -1).detach().cpu().numpy()
            res_pred = [seq2res(x) for x in pred]
            acc = np.mean(np.array(res_pred) == res.numpy())
            train_acc.append(acc)
        train_acc = np.mean(train_acc)
        train_loss = np.mean(train_loss)
        print("Train acc: %.2f, loss: %.3f "%(train_acc * 100, train_loss))
            
        if ((epoch+1) % args.epochs_eval == 0) or (epoch+1 == args.epochs):
            perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader)
            print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('val', 100*perception_acc, 100*head_acc, 100*result_acc))
            if result_acc > best_acc:
                best_acc = result_acc

            # model_path = args.output_dir + "model_%03d.p"%(epoch + 1)
            # model.save(model_path, epoch=epoch+1)
                
        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    # Test
    print('-' * 30)
    print('Evaluate on test set...')
    eval_dataloader = torch.utils.data.DataLoader(args.test_set, batch_size=64,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader)
    print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('test', 100*perception_acc, 100*head_acc, 100*result_acc))
    return



if __name__ == "__main__":
    args = parse_args()
    sys.argv = sys.argv[:1]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)

    # train_set = HINT('train', numSamples=5000)
    train_set = HINT('train')
    val_set = HINT('val')
    # test_set = HINT('val')
    test_set = HINT('test')
    print('train:', len(train_set), 'val:', len(val_set), 'test:', len(test_set))

    model = make_model(args)
    model.to(DEVICE)

    if args.perception_pretrain and not args.perception:
        model.embedding_in.image_encoder.load_state_dict(torch.load(args.perception_pretrain))

    st_epoch = 0
    if args.resume:
        st_epoch = model.load(args.resume)
        if st_epoch is None:
            st_epoch = 0


    print(args)
    print(model)
    args.train_set = train_set
    args.val_set = val_set
    args.test_set = test_set

    train(model, args, st_epoch=st_epoch)

