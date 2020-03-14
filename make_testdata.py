import argparse
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='cat', help='category name for use')
parser.add_argument('--output', default='train_files', help='output directory')
parser.add_argument('--maxsize', default=-1, type=int, help='max num images')
parser.add_argument('--imagesize', default=256, type=int, help='output image size')
args = parser.parse_args()

dataDir ='.'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

if not os.path.isdir(args.output):
    os.mkdir(args.output)
    os.mkdir(args.output+'/imgs')
    os.mkdir(args.output+'/mask')

coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
catnames = [args.category] if not ',' in args.category else args.category.split(',')

catIds = coco.getCatIds(catNms=catnames)
imgIds = coco.getImgIds(catIds=catIds)
imgIds = imgIds[:args.maxsize] if args.maxsize>0 else imgIds

for id in imgIds:
    img = coco.loadImgs([id])[0]
    annIds = coco.getAnnIds(imgIds=[id], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = None
    for ann in anns:
        if mask is None:
            mask = coco.annToMask(ann)
        else:
            mask[coco.annToMask(ann) != 0] = 1
    mask = mask.astype(np.uint8) * 255
    mask = Image.fromarray(mask).resize((args.imagesize,args.imagesize))
    mask = mask.convert("L")

    imgFile = '%s/%s/%s'%(dataDir,dataType,img['file_name'])
    image = Image.open(imgFile)
    image = image.convert("RGB")
    image = image.resize((args.imagesize,args.imagesize))
    image.save(args.output+'/imgs/%d.png'%id)
    mask.save(args.output+'/mask/%d.png'%id)
