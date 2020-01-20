Segmentation
------------


Based on FAIR detecton2
```sh
$ python detector.py -h
usage: detector.py [-h] [-i IMAGE] [-c COCO] [-v VIDEO] [-o OUTPUT] [-m MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Image or image directory
  -c COCO, --coco COCO  Coco image category
  -v VIDEO, --video VIDEO
                        Input video file
  -o OUTPUT, --output OUTPUT
                        Output video file
  -m MODEL, --model MODEL
                        Detectron pretrained segmentation model, e.g.
                        instance/object/keypoint/panoptic, COCO-
                        InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

```

ffmpeg cut video
```sh
# -t duration
ffmpeg -i video.mp4 -t 00:00:06 -c:v copy video-clip.mp4

# -ss start time offset
ffmpeg -i video.mp4 -ss 00:03:36 -t 00:00:60 -c:v copy video-clip.mp4

```