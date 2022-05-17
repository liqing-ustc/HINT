from utils import SYM2ID, ROOT_DIR, IMG_DIR, NULL, IMG_TRANSFORM, pad_image
from copy import deepcopy
import random
import json
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

class HINT(Dataset):
    def __init__(self, split='train', exclude_symbols=None, max_len=None, numSamples=None, fewshot=-1):
        super(HINT, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.dataset = json.load(open(ROOT_DIR + ('fewshot_%d_'%fewshot if fewshot !=-1 else '') + 'expr_%s.json'%split))
        if numSamples:
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:numSamples]
        
        if exclude_symbols is not None:
            exclude_symbols = set(exclude_symbols)
            self.dataset = [x for x in self.dataset if len(set(x['expr']) & exclude_symbols) == 0]

        if max_len is not None:
            self.dataset = [x for x in self.dataset if len(x['expr']) <= max_len]
            
        for x in self.dataset:
            x['len'] = len(x['expr'])
        
        self.img_transform = IMG_TRANSFORM
        self.valid_ids = list(range(len(self.dataset)))

        # dataset statistics, used to filter samples
        len2ids = {}
        for i, x in enumerate(self.dataset):
            l = len(x['img_paths'])
            if l not in len2ids:
                len2ids[l] = []
            len2ids[l].append(i)
        self.len2ids = len2ids

        sym2ids = {}
        for i, x in enumerate(self.dataset):
            for s in list(set(x['expr'])):
                if s not in sym2ids:
                    sym2ids[s] = []
                sym2ids[s].append(i)
        self.sym2ids = sym2ids

        res2ids = {}
        for i, x in enumerate(self.dataset):
            l = x['res']
            if l not in res2ids:
                res2ids[l] = []
            res2ids[l].append(i)
        self.res2ids = res2ids

        digit2ids = {}
        for i, x in enumerate(self.dataset):
            if len(x['expr']) == 1:
                s = x['expr'][0]
                if s not in digit2ids:
                    digit2ids[s] = []
                digit2ids[s].append(i)
        self.digit2ids = digit2ids

        if split in ['val', 'test']:
            cond2ids = {i: [] for i in range(1, 6)}
            for i, x in enumerate(self.dataset):
                cond2ids[x['eval']].append(i)
            self.cond2ids = cond2ids

    def __getitem__(self, index):
        index = self.valid_ids[index]
        sample = deepcopy(self.dataset[index])
        img_seq = []
        for img_path in sample['img_paths']:
            img = Image.open(IMG_DIR+img_path).convert('L')
            img = ImageOps.invert(img)
            img = pad_image(img, 60)
            img = transforms.functional.resize(img, 40)
            img = self.img_transform(img)
            img_seq.append(img)
        # del sample['img_paths']
        sample['expr'] = ''.join(sample['expr'])
        
        sentence = [SYM2ID(sym) for sym in sample['expr']]
        sample['img_seq'] = img_seq
        sample['sentence'] = sentence
        return sample
            
    
    def __len__(self):
        return len(self.valid_ids)

    def filter_by_len(self, min_len=None, max_len=None):
        if min_len is None: min_len = -1
        if max_len is None: max_len = float('inf')
        self.valid_ids = [i for i, x in enumerate(self.dataset) if x['len'] <= max_len and x['len'] >= min_len]
    
    def filter_by_eval(self, eval_idx=None):
        if eval_idx is None:
            self.valid_ids = list(range(len(self.dataset)))
        else:
            self.valid_ids = self.cond2ids[eval_idx]

    def all_symbols(self, max_len=float('inf')):
        dataset = [sample for sample in self.dataset if len(sample['expr']) <= max_len]
        symbol_set = [(x,SYM2ID(y)) for sample in dataset for x, y in zip(sample['img_paths'], sample['expr'])]
        return sorted(list(symbol_set))

def HINT_collate(batch):
    img_seq_list = []
    sentence_list = []
    img_paths_list = []
    head_list = []
    res_all_list = []
    for sample in batch:
        img_seq_list.extend(sample['img_seq'])
        del sample['img_seq']

        img_paths_list.append(sample['img_paths'])
        del sample['img_paths']

        sentence_list.append(sample['sentence'])
        del sample['sentence']

        head_list.append(sample['head'])
        del sample['head']

        res_all_list.append(sample['res_all'])
        del sample['res_all']
        
    batch = default_collate(batch)
    batch['img_seq'] = torch.stack(img_seq_list)
    batch['img_paths'] = img_paths_list
    batch['sentence'] = sentence_list
    batch['head'] = head_list
    batch['res_all'] = res_all_list
    return batch

if __name__ == '__main__':
    val_set = HINT('val')
    val_loader = DataLoader(val_set, batch_size=32,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)

    print(next(iter(val_loader)))