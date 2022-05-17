import sys
from data.domain import *
import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = './data/'
IMG_DIR = ROOT_DIR + 'symbol_images/'
IMG_SIZE = 32

from torchvision import transforms
IMG_TRANSFORM = transforms.Compose([
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: 1. - x),
                    # transforms.Normalize((0.5,), (1,))
                ])

from PIL import Image, ImageOps
def pad_image(img, desired_size, fill=0):
    delta_w = desired_size - img.size[0]
    delta_h = desired_size - img.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_img = ImageOps.expand(img, padding, fill)
    return new_img

INP_VOCAB = SYMBOLS + [START, END, NULL]
RES_VOCAB = DIGITS + [START, END, NULL]

RES_MAX_LEN = 10

reverse = True
def res2seq(res, pad=True):
    seq = [list(str(r)) for r in res]
    if reverse:
        seq = [s[::-1] for s in seq]
    seq = [[START] + s + [END] for s in seq]
    if pad:
        max_len = max([len(s) for s in seq])
        seq = [s + [NULL]*(max_len - len(s)) for s in seq]
    seq = [list(map(RES_VOCAB.index, s)) for s in seq]
    return seq

def seq2res(seq):
    res = []
    for x in seq:
        x = RES_VOCAB[x]
        if x in [START, NULL, END]:
            break
        res.append(x)

    if reverse:
        res = res[::-1]
    res = int(''.join(res)) if len(res) > 0 else -1
    return res
