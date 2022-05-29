import sys
from data.domain import *
import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = './data/'
IMG_DIR = ROOT_DIR + 'expr_images/'
IMG_SIZE = 32 # original image for each symbol is (45, 45)

from torchvision import transforms
IMG_TRANSFORM = transforms.Compose([
                    transforms.Resize(IMG_SIZE),
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

symbol_images_dir = ROOT_DIR + 'symbol_images/'
def render_img(img_paths):
    images = [Image.open(symbol_images_dir + x) for x in img_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('L', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

INP_VOCAB = SYMBOLS + [START, END, NULL]