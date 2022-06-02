from utils import SYM2ID, ROOT_DIR, IMG_DIR, NULL, IMG_TRANSFORM, pad_image, IMG_SIZE, render_img
from utils import OPERATORS, FEWSHOT_OPERATORS
from copy import deepcopy
import random
import json
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

op_list =  OPERATORS + FEWSHOT_OPERATORS
def expr2n_op(expr):
    return len([1 for x in expr if x in op_list])

class HINT(Dataset):
    def __init__(self, split, input, fewshot=None, n_sample=None, max_op=None, main_dataset_ratio=0.):
        super(HINT, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.input = input
        self.fewshot = fewshot

        if fewshot:
            dataset = json.load(open(ROOT_DIR + 'fewshot_dataset.json'))
            dataset = dataset[fewshot]
            dataset = dataset[split]
            self.main_dataset_ratio = main_dataset_ratio
            if split == 'train' and main_dataset_ratio > 0:
                self.main_dataset = json.load(open(ROOT_DIR + 'expr_%s.json'%split))
        else:
            dataset = json.load(open(ROOT_DIR + 'expr_%s.json'%split))

        if n_sample:
            if n_sample <= 1: # it is percentage
                n_sample = int(len(dataset) * n_sample)
            random.shuffle(dataset)
            dataset = dataset[:n_sample]
            print(f'{split}: randomly select {n_sample} samples.')
            
        if isinstance(max_op, int):
            dataset = [x for x in dataset if expr2n_op(x['expr']) <= max_op]
            print(f'{split}: filter {len(dataset)} samples with no more than {max_op} operators.')

        for x in dataset:
            x['len'] = len(x['expr'])
        
        self.dataset = dataset
        self.img_transform = IMG_TRANSFORM
        self.valid_ids = list(range(len(dataset)))

    @property
    def max_dep2ids(self):
        """max dependency distance."""
        if hasattr(self, '_max_dep2ids'):
            return self._max_dep2ids
        else:
            def compute_max_dep(heads):
                return max([0] + [abs(i-h) for i, h in enumerate(heads) if h != -1])

            def sample2key(sample):
                return compute_max_dep(sample['head'])

            mapping = {}
            for i, x in enumerate(self.dataset):
                k = sample2key(x)
                if k not in mapping:
                    mapping[k] = []
                mapping[k].append(i)
            self._max_dep2ids = mapping
            return mapping

    @property
    def ps_depth2ids(self):
        """parenthesis depth."""
        if hasattr(self, '_ps_depth2ids'):
            return self._ps_depth2ids
        else:
            lps = '('
            rps = ')'
            def compute_ps_depth(expr):
                depth = 0
                max_depth = 0
                for x in expr:
                    if x == lps:
                        c = 1
                    elif x == rps:
                        c = -1
                    else:
                        c = 0
                    depth += c
                    if depth > max_depth:
                        max_depth = depth
                return max_depth

            def sample2key(sample):
                return compute_ps_depth(sample['expr'])

            mapping = {}
            for i, x in enumerate(self.dataset):
                k = sample2key(x)
                if k not in mapping:
                    mapping[k] = []
                mapping[k].append(i)
            self._ps_depth2ids = mapping
            return mapping


    @property
    def tree_depth2ids(self):
        if hasattr(self, '_tree_depth2ids'):
            return self._tree_depth2ids
        else:
            from functools import lru_cache
            def compute_tree_depth(head):
                @lru_cache
                def depth(i):
                    """The depth of node i."""
                    if head[i] == -1:
                        return 1
                    return depth(head[i]) + 1
                
                return max(depth(i) for i in range(len(head)))

            def sample2key(sample):
                return compute_tree_depth(sample['head'])

            mapping = {}
            for i, x in enumerate(self.dataset):
                k = sample2key(x)
                if k not in mapping:
                    mapping[k] = []
                mapping[k].append(i)
            self._tree_depth2ids = mapping
            return mapping
    
    @property
    def eval2ids(self):
        if hasattr(self, '_eval2ids'):
            return self._eval2ids
        else:
            def sample2key(sample):
                return sample['eval']

            mapping = {}
            for i, x in enumerate(self.dataset):
                k = sample2key(x)
                if k not in mapping:
                    mapping[k] = []
                mapping[k].append(i)
            self._eval2ids = mapping
            return mapping

    @property
    def digit2ids(self):
        if hasattr(self, '_digit2ids'):
            return self._digit2ids
        else:
            def sample2key(sample):
                if len(sample['expr']) == 1:
                    return sample['expr'][0]
                return None

            mapping = {}
            for i, x in enumerate(self.dataset):
                k = sample2key(x)
                if not k:
                    continue
                if k not in mapping:
                    mapping[k] = []
                mapping[k].append(i)
            self._digit2ids = mapping
            return mapping

    @property
    def result2ids(self):
        if hasattr(self, '_result2ids'):
            return self._result2ids
        else:
            def sample2key(sample):
                r = sample['res']
                if r < 10:
                    return r
                r = (r // 10) * 10
                r = min(r, 100)
                return r

            mapping = {}
            for i, x in enumerate(self.dataset):
                k = sample2key(x)
                if k not in mapping:
                    mapping[k] = []
                mapping[k].append(i)
            self._result2ids = mapping
            return mapping

    @property
    def length2ids(self):
        if hasattr(self, '_length2ids'):
            return self._length2ids
        else:
            def sample2key(sample):
                return len(sample['img_paths'])

            mapping = {}
            for i, x in enumerate(self.dataset):
                k = sample2key(x)
                if k not in mapping:
                    mapping[k] = []
                mapping[k].append(i)
            self._length2ids = mapping
            return mapping

    @property
    def symbol2ids(self):
        if hasattr(self, '_symbol2ids'):
            return self._symbol2ids
        else:
            def sample2key(sample):
                return list(set(sample['expr']))

            mapping = {}
            for i, x in enumerate(self.dataset):
                k_list = sample2key(x)
                for k in k_list:
                    if k not in mapping:
                        mapping[k] = []
                    mapping[k].append(i)
            self._symbol2ids = mapping
            return mapping

    def __getitem__(self, index):
        if self.fewshot and self.split == 'train' and random.random() < self.main_dataset_ratio:
            # use sample from main dataset to avoid forgetting
            sample = random.choice(self.main_dataset)
            sample = deepcopy(sample)
        else:
            index = self.valid_ids[index]
            sample = deepcopy(self.dataset[index])
        if self.input == 'image':
            # img_path = sample['expr_image']
            # img = Image.open(IMG_DIR+img_path).convert('L')
            img = render_img(sample['img_paths'])
            img = ImageOps.invert(img)
            img = self.img_transform(img)
            img = torch.cat([img, torch.zeros((2, *img.shape[1:]))])

            img_seq = transform_img_seq(img, IMG_SIZE, IMG_SIZE)
            sample['img_seq'] = img_seq
            sample['len'] = len(img_seq)
        # del sample['img_paths']
        sample['expr'] = ''.join(sample['expr'])
        
        sentence = [SYM2ID(sym) for sym in sample['expr']]
        sample['sentence'] = sentence
        return sample
    
    def __len__(self):
        return len(self.valid_ids)

    def filter_by_len(self, min_len=None, max_len=None):
        if min_len is None: min_len = -1
        if max_len is None: max_len = float('inf')
        self.valid_ids = [i for i, x in enumerate(self.dataset) if x['len'] <= max_len and x['len'] >= min_len]
    

    def all_symbols(self, max_len=float('inf')):
        dataset = [sample for sample in self.dataset if len(sample['expr']) <= max_len]
        symbol_set = [(x,SYM2ID(y)) for sample in dataset for x, y in zip(sample['img_paths'], sample['expr'])]
        return sorted(list(symbol_set))

def transform_img_seq(expr_img, window_size, stride):
    slide_dim = 2
    st = 0
    img_seq = []
    while st + window_size <= expr_img.shape[slide_dim]:
        indices = torch.arange(st, st+window_size)
        img = torch.index_select(expr_img, slide_dim, indices)
        img_seq.append(img)
        st += stride
    return img_seq

def HINT_collate(batch):
    img_seq_list = []
    sentence_list = []
    img_paths_list = []
    head_list = []
    res_all_list = []
    for sample in batch:
        if 'img_seq' in sample:
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
    if img_seq_list:
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