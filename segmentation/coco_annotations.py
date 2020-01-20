import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random
import skimage
import sys

input_json = sys.argv[1]
output_json = sys.argv[2]

with open(input_json, 'r') as f:
    anno = json.loads(f.read())


image_mapping = {}
for image in anno['images']:
    image_mapping[str(image['id'])] = image

cats = anno['categories']
cat_mapping = {}
for cat in cats:
    cat_mapping[cat['id']] = {'name': cat['name'], 'supercategory': cat['supercategory']}


ann_mapping = {}
for a in anno['annotations']:
    idx = str(a['image_id'])
    if idx in ann_mapping:
        continue
    ann_mapping[idx] = a


mapping = {}
for key in image_mapping.keys():
    new_dict = {}
    ann_dict = ann_mapping[key]

    category_id = ann_dict['category_id']
    cat = cat_mapping[category_id]
    ann_dict['category'] = cat['name']
    ann_dict['supercategory'] = cat['supercategory']

    image_dict = image_mapping[key]
    file_name = image_dict['file_name']

    new_dict = image_dict
    new_dict.update(ann_dict)
    del new_dict['segmentation']
    mapping[file_name] = new_dict


with open(output_json, 'w') as f:
    js = json.dumps(mapping)
    f.write(js)