"""
fork from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
"""
from pycocotools.coco import COCO
import random
import skimage
import matplotlib.pyplot as plt
import os


class Cocoapi():

    def __init__(self,
            coco_path='/home/ubuntu/dataset/coco',
            subset='val2017'):
        self.coco_path = coco_path
        self.subset = subset
        ann_json_path = os.path.join(coco_path, 'annotations', f'{subset}.json')
        self.coco = COCO(ann_json_path)

    def cats(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        names = [cat['name'] for cat in cats]
        names_super = set([cat['supercategory'] for cat in cats])
        names.extend(names_super)
        return names

    def info(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

    def get(self, categories=['person']):
        catIds = self.coco.getCatIds(catNms=categories)
        imgIds = self.coco.getImgIds(catIds=catIds)
        img_obj = self.coco.loadImgs(random.choice(imgIds))[0]
        image_path = os.path.join(self.coco_path, self.subset, img_obj['file_name'])
        if os.path.isfile(image_path):
            img = image_path
        else:
            img = img_obj['coco_url']
        print(img)
        return img

    def display(self, img_path):
        img = skimage.io.imread(img_path)
        plt.axis('off')
        plt.imshow(img)
        plt.show()