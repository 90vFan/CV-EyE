import detectron2
from detectron2.utils.logger import setup_logger

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import torch
import skimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import imutils
import argparse
import time

from coco_api import Cocoapi

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='Image or image directory')
parser.add_argument('-c', '--coco', help='Coco image category')
parser.add_argument(
    '-m', '--model', required=False,
    default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    help='Detectron pretrained segmentation model, e.g. instance/object/keypoint/panoptic/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
args = vars(parser.parse_args())
ARG_IMAGE = args['image']
ARG_MODEL = args['model']
ARG_COCO = args['coco']

MODEL_CONFIG_MAPPING = {
    'instance': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'object': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    'keypoint': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
    'panoptic': 'COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
}
MODEL_CONFIG = ARG_MODEL if ARG_MODEL.endswith('yaml') else MODEL_CONFIG_MAPPING[ARG_MODEL]

logger = setup_logger()


class Detector(object):

    def __init__(self):
        self.arg_image = ARG_IMAGE
        self.arg_coco = ARG_COCO
        self.metadata = None
        self.cpu_device = torch.device("cpu")
        self.cfg = self._init_cfg()
        self.predictor = self._init_predictor()

    def get_coco_image(self):
        logger.info('get image form coco dataset ...')
        coco = Cocoapi()
        cats = coco.cats()
        logger.info(f'input coco category is {self.arg_coco}')
        if self.arg_coco in cats:
            image_path = coco.get(self.arg_coco.split(','))
        else:
            logger.error(f'coco category includes: {cats}')
            raise KeyError('Input argument "--coco" is not belong to coco categories')

        return image_path

    def _init_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        return cfg

    def _init_predictor(self):
        predictor = DefaultPredictor(self.cfg)
        return predictor

    def parse_image_path(self):
        if os.path.isdir(self.arg_image):
            image_paths = next(os.walk(self.arg_image))[2]
            image_name = random.choice(image_paths)
            image_path = os.path.join(self.arg_image, image_name)
        elif os.path.isfile(self.arg_image):
            image_path = self.arg_image
        else:
            raise KeyError('Input argument "--image" is not an image or image directory')
        logger.info(f'image path: {image_path}')
        return image_path

    def read_image(self, image_path):
        logger.info('loading image ...')
        im = skimage.io.imread(image_path)
        im = imutils.resize(im, width=800)
        return im

    def predict(self, im):
        logger.info('predicting ...')
        predictions = self.predictor(im)
        return predictions

    def draw_segmentation(self, im, predictions):
        visualizer = Visualizer(im, self.metadata)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                box_count = len(predictions['instances'].pred_classes)
                logger.info(f'detected {box_count} instances')

        return vis_output

    def display_images(self, im, v):
        cv2.imshow('Origin Image', im[:, :, ::-1])
        cv2.moveWindow('Origin Image', 50, 0)
        seg_image = v.get_image()[:, :, ::-1]
        cv2.imshow('Segmetation Image', imutils.resize(seg_image, 800))
        cv2.moveWindow('Segmetation Image', im.shape[1] + 50, 0)
        cv2.waitKey(0) & 0xFF == ord('q')
        logger.info('closed')
        cv2.destroyAllWindows()

    def run(self):
        if ARG_IMAGE:
            image_path = self.parse_image_path()
        elif ARG_COCO:
            image_path = self.get_coco_image()
        im = self.read_image(image_path)

        start = time.time()
        predictions = self.predict(im)
        v = self.draw_segmentation(im, predictions)
        end = time.time()
        duration = '{0:.3f}'.format(end - start)
        logger.info(f'takes {duration} seonds')

        self.display_images(im, v)


if __name__ == "__main__":
    detector = Detector()
    detector.run()
