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
    def __init__(self, split='train', input='image', exclude_symbols=None, max_len=None, numSamples=None, fewshot=-1):
        super(HINT, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.input = input
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
        index = self.valid_ids[index]
        sample = deepcopy(self.dataset[index])
        if self.input == 'image':
            img_seq = []
            for img_path in sample['img_paths']:
                img = Image.open(IMG_DIR+img_path).convert('L')
                img = ImageOps.invert(img)
                img = pad_image(img, 60)
                img = transforms.functional.resize(img, 40)
                img = self.img_transform(img)
                img_seq.append(img)
            sample['img_seq'] = img_seq
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