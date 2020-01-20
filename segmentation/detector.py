import detectron2
from detectron2.utils.logger import setup_logger

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
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
import tqdm

from coco_api import Cocoapi

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='Image or image directory')
parser.add_argument('-c', '--coco', help='Coco image category')
parser.add_argument('-v', '--video', help='Input video file')
parser.add_argument('-o', '--output', help='Output video file')
parser.add_argument(
    '-m', '--model', required=False,
    default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    help='Detectron pretrained segmentation model, e.g. instance/object/keypoint/panoptic, ' + \
         'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
args = parser.parse_args()
ARG_MODEL = args.model

MODEL_CONFIG_MAPPING = {
    'instance': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'object': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    'keypoint': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
    'panoptic': 'COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
}
MODEL_CONFIG = ARG_MODEL if ARG_MODEL.endswith('yaml') else MODEL_CONFIG_MAPPING[ARG_MODEL]

logger = setup_logger()


class Detector(object):

    def __init__(self, instance_mode=ColorMode.IMAGE):
        self.metadata = None
        self.instance_model = instance_mode
        self.cpu_device = torch.device("cpu")
        self.cfg = self._init_cfg()
        self.predictor = self._init_predictor()

    def get_coco_image(self):
        logger.info('get image form coco dataset ...')
        coco = Cocoapi()
        cats = coco.cats()
        logger.info(f'input coco category is {args.coco}')
        if args.coco in cats:
            image_path = coco.get(args.coco.split(','))
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
        if os.path.isdir(args.image):
            image_paths = next(os.walk(args.image))[2]
            image_name = random.choice(image_paths)
            image_path = os.path.join(args.image, image_name)
        elif os.path.isfile(args.image):
            image_path = args.image
        else:
            raise KeyError('Input argument "--image" is not an image or image directory')
        logger.info(f'image path: {image_path}')
        return image_path

    def run_on_image(self):
        if args.image:
            image_path = self.parse_image_path()
        elif args.coco:
            image_path = self.get_coco_image()
        im = self.read_image(image_path)

        start = time.time()
        predictions = self.predict(im)
        vis_image = self.draw_segmentation(im, predictions)
        end = time.time()
        duration = '{0:.3f}'.format(end - start)
        logger.info(f'takes {duration} seonds')

        self.display_images(im, vis_image)

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
        visualizer = Visualizer(im, self.metadata, instance_model=self.instance_model)

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

        vis_image = vis_output.get_image()[:, :, ::-1]
        return vis_image

    def display_images(self, im, vis_image):
        cv2.imshow('Origin Image', im[:, :, ::-1])
        cv2.moveWindow('Origin Image', 50, 0)
        cv2.imshow('Segmetation Image', imutils.resize(vis_image, 800))
        cv2.moveWindow('Segmetation Image', im.shape[1] + 50, 0)
        cv2.waitKey(0) & 0xFF == ord('q')
        logger.info('closed')
        cv2.destroyAllWindows()

    def convert_video(self):
        video = cv2.VideoCapture(args.video)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video)

        if args.output:
            output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            video_writer = cv2.VideoWriter(
                filename=output_fname,
                fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video)
        for vis_frame in tqdm.tqdm(self.run_on_video(video), total=num_frames):
            if args.output:
                video_writer.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # esc to quit

        video.release()
        if args.output:
            video_writer.release()
        else:
            cv2.destroyAllWindows()

    def run_on_video(self, video):
        video_visualizer = VideoVisualizer(self.metadata, self.instance_model)
        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield self.process_predictions(frame, video_visualizer, self.predictor(frame))

    def process_predictions(self, frame, video_visualizer, predictions):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                frame, panoptic_seg.to(self.cpu_device), segments_info
            )
        elif "instances" in predictions:
            predictions = predictions["instances"].to(self.cpu_device)
            vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
        elif "sem_seg" in predictions:
            vis_frame = video_visualizer.draw_sem_seg(
                frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
            )

        # Converts Matplotlib RGB format to OpenCV BGR format
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        return vis_frame

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run(self):
        if args.image:
            self.run_on_image()
        elif args.video:
            self.convert_video()


if __name__ == "__main__":
    detector = Detector()
    detector.run()
