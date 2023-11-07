import os 
import argparse

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

import json


parser = argparse.ArgumentParser()

parser.add_argument("dataset_path", help="File path cityscapes datasets folder", type=str)
parser.add_argument("--dataset_type", default='cityscapes', type=str)
parser.add_argument("--opacity", default=0.5, type=float)


args = parser.parse_args()

def blendImageWithColorMap(image, label, palette, alpha):
    
    color_map = np.zeros_like(image)
        
    for idx, color in enumerate(palette) : 
        
        if idx == 0 :
            color_map[label == idx, :] = image[label == idx, :] * 1
        else :
            color_map[label == idx, :] = image[label == idx, :] * alpha + color * (1-alpha)

    return color_map


def imread(path):
    
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    nparray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)

    return bgrImage

def imwrite(path, image):
    _, ext = os.path.splitext(path)
    cv2.imencode(ext, image)[1].tofile(path)

def main():

    source_path = args.dataset_path
    dataset_type = args.dataset_type
    alpha = args.opacity

    hdr_path = glob(os.path.join(source_path, "*.hdr"))
    hdr_path = hdr_path[0]

    print(hdr_path)
    
    with open(hdr_path) as f:
        hdr = json.load(f)

    palette = []
    
    for idx, cat in enumerate(hdr['categories']):
        name, color = cat[0], cat[1]
        color = json.loads(color)
        r_ch = color[0]
        b_ch = color[2]
        color[0] = b_ch
        color[2] = r_ch
        palette.append(color)

    palette = np.array(palette)


    
    if dataset_type == 'cityscapes' : 

        img_list = glob(os.path.join(source_path, 'leftImg8bit', '*', '*.png'))

        if not img_list :
            raise Exception("There is no image in the leftImg8bit folder.")

    # loop through all the imgs 
    for img_path in tqdm(img_list) :
        img = imread(img_path)
        
        gt_img_path = img_path.replace('\\leftImg8bit', '\\gtFine')
        gt_img_path = gt_img_path.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        gt_img = imread(gt_img_path)
        
            
        color_path = img_path.replace('\\leftImg8bit', '\\color')
        color_path = color_path.replace('_leftImg8bit.png', '_color.png')
        colormap = blendImageWithColorMap(image=img, 
                                            label=gt_img, 
                                            palette=palette, 
                                            alpha=alpha)
        
        os.makedirs(os.path.dirname(color_path), exist_ok=True)
        
        imwrite(color_path, colormap)
        
if __name__ == '__main__':
    

    main()
