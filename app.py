
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import sys
import datetime
from ultralytics import YOLO 
from PIL import Image
model = YOLO("cup_detector.pt")
source = "/kaggle/input/video-weights/dulux_paint.mp4"
results = model.track(source = source)

rbox = []
threshold = 0.30
for r in results:
    if r is None:
        rbox.append(np.asarray([[-1,-1,-1,-1]]).astype('int'))
    elif len(r.boxes.conf.tolist()) < 1:
        rbox.append(np.asarray([[-1,-1,-1,-1]]).astype('int'))
    elif r.boxes.conf.tolist()[0] < threshold:
        rbox.append(np.asarray([[-1,-1,-1,-1]]).astype('int'))
    else :
        rbox.append(np.round(np.asarray(r.boxes.xyxy.tolist())).astype('int'))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-26T12:57:38.594582Z","iopub.execute_input":"2023-08-26T12:57:38.595183Z","iopub.status.idle":"2023-08-26T12:57:38.650460Z","shell.execute_reply.started":"2023-08-26T12:57:38.595140Z","shell.execute_reply":"2023-08-26T12:57:38.648993Z"},"jupyter":{"outputs_hidden":false}}
no_of_frames = 1000 # Number of tracking boxes that will be smoothened out. For a still image and object include all the frames
smooth = 1 # The degree of smoothness. 
k = 0
while True:
    boxes = rbox[k*no_of_frames:k*no_of_frames+no_of_frames]
    if len(boxes) == 0:
        break
    x1 = [arr[0][0] for arr in boxes if arr[0][0] != -1]
    x1_mean = np.mean(x1)
    y1 = [arr[0][1] for arr in boxes if arr[0][1] != -1 ]
    y1_mean = np.mean(y1)
    x2 = [arr[0][2] for arr in boxes if arr[0][2] != -1 ]
    x2_mean = np.mean(x2)
    y2 = [arr[0][3] for arr in boxes if arr[0][3] != -1]
    y2_mean = np.mean(y2)
    for i in range(len(boxes)):
        if boxes[i][0][0] < 0:
            #log(boxes[i][0][0], end = " from -1 \n")
            pass
        else:
            rbox[k*no_of_frames:k*no_of_frames+no_of_frames][i][0][0] = np.round(boxes[i][0][0] + (x1_mean - boxes[i][0][0])*smooth)
            rbox[k*no_of_frames:k*no_of_frames+no_of_frames][i][0][1] = np.round(boxes[i][0][1] + (y1_mean - boxes[i][0][1])*smooth)
            rbox[k*no_of_frames:k*no_of_frames+no_of_frames][i][0][2] = np.round(boxes[i][0][2] + (x2_mean - boxes[i][0][2])*smooth)
            rbox[k*no_of_frames:k*no_of_frames+no_of_frames][i][0][3] = np.round(boxes[i][0][3] + (y2_mean - boxes[i][0][3])*smooth)
    k = k + 1






# Provide the path to your image
image_path = '/kaggle/input/video-weights/nescafe_cup.png'


cv2.imread("/kaggle/input/video-weights/nescafe_cup.png")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-26T12:57:51.200890Z","iopub.execute_input":"2023-08-26T12:57:51.201260Z","iopub.status.idle":"2023-08-26T12:57:51.205886Z","shell.execute_reply.started":"2023-08-26T12:57:51.201229Z","shell.execute_reply":"2023-08-26T12:57:51.204718Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt

# Provide the path to your RGB image, the color to increase ('red', 'green', or 'blue'), and the increase factor
image_path = '/kaggle/input/video-weights/nescafe_cup.png'

video_path = source # can be found in the data link will be shared if you want to test out your own
ad_image_path = '/kaggle/input/video-weights/mcdonalds.png' # Can be found in the data
bounding_boxes = rbox

# Load the video using OpenCV
video_cap = cv2.VideoCapture(video_path)

# Load the ad image and check if it's loaded successfully
ad_image = cv2.imread(ad_image_path , cv2.IMREAD_UNCHANGED)

if ad_image is None:
    raise ValueError("Failed to load the ad image. Please check the file path and format.")

# Get video properties
fps = video_cap.get(cv2.CAP_PROP_FPS)
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create VideoWriter object to save the final video with the ad overlay
output_path = 'demo_cup.mp4' # output video naame

import os
import cv2
import argparse
import numpy as np
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf
import sys
from pathlib import Path

print("path")
print(sys.path)
from src import model

ad_image_path = "/kaggle/input/video-weights/mcdonalds.png"
bg_video_path = source

def load_video_frames(video_path):
    frames = []
    
    vc = cv2.VideoCapture(video_path)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
         rval = False
    if not rval:
        return frames
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    for fdx in range(0, int(frame_num)):
        frames.append(frame)
        rval, frame = vc.read()
    
    return frames,fps


# pre-defined arguments
cuda = torch.cuda.is_available()

# create/load the harmonizer model
print('Create/load Harmonizer...\n')
harmonizer = model.Harmonizer()
if cuda:
    harmonizer = harmonizer.cuda()
harmonizer.load_state_dict(torch.load('./pretrained/harmonizer.pth'), strict=True)
harmonizer.eval()


# define example paths
harmonized_video_path = "demo_cup.mp4"


# read input videos
bg_frames, fps = load_video_frames(bg_video_path)
fps = 25
ema = 1 - 1 / fps

image_data = cv2.imread(ad_image_path , cv2.IMREAD_UNCHANGED)

h, w = bg_frames[0].shape[:2]

# define video writer

h, w = bg_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
harmonized_vw = cv2.VideoWriter(harmonized_video_path, fourcc, fps, (w, h))

# harmonization
ema_arguments = None
pbar = tqdm(range(len(bg_frames)), total=len(bg_frames), unit='frame')

for i, fdx in enumerate(pbar):
    x,y,w,h = 0 ,0 ,0,0,
    for box in bounding_boxes[i]:
        if len(box) < 4:
            continue
        if box[0] < 0 or box[1] < 0 or box[2] <0 or box[3] < 0:
            continue
        x, y, xmax, ymax = box
        w = int((xmax - x))
        h = int((ymax - y))
    height, width = bg_frames[0].shape[:2]
    image = Image.fromarray(image_data)
    iamge = image.transpose(Image.FLIP_LEFT_RIGHT)
    image.thumbnail((w+60, h+40))
    image = np.asarray(image.convert("RGBA"))
    new = Image.new(mode="RGBA", size=(width,height))
    new.paste(Image.fromarray(image), (x,y), Image.fromarray(image))
    mask_frames = new.split()[3]
    mask_frames = np.asarray(mask_frames.convert("RGB"))
    fg_frames = np.asarray(new.convert("RGB"))
    mask = cv2.cvtColor(mask_frames, cv2.COLOR_BGR2RGB)
    fg = cv2.cvtColor(fg_frames, cv2.COLOR_BGR2RGB)
    bg = cv2.cvtColor(bg_frames[fdx % len(bg_frames)], cv2.COLOR_BGR2RGB)

    comp = fg * (mask / 255.0) + bg * (1 - mask / 255.0)

    comp = Image.fromarray(comp.astype(np.uint8))
    mask = Image.fromarray(mask[:, :, 0].astype(np.uint8))

    comp = tf.to_tensor(comp)[None, ...]
    mask = tf.to_tensor(mask)[None, ...]

    if cuda:
        comp = comp.cuda()
        mask = mask.cuda()

    with torch.no_grad():
        arguments = harmonizer.predict_arguments(comp, mask)

        if ema_arguments is None:
            ema_arguments = list(arguments)
        else:
            for i, (ema_argument, argument) in enumerate(zip(ema_arguments, arguments)):
                ema_arguments[i] = ema * ema_argument + (1 - ema) * argument

        harmonized = harmonizer.restore_image(comp, mask, ema_arguments)[-1]

    comp = np.transpose(comp[0].cpu().numpy(), (1, 2, 0)) * 255
    comp = cv2.cvtColor(comp.astype('uint8'), cv2.COLOR_RGB2BGR)

    harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
    harmonized = cv2.cvtColor(harmonized.astype('uint8'), cv2.COLOR_RGB2BGR)
    harmonized_vw.write(harmonized)

harmonized_vw.release()

print('\n')

print('Finished.')
print('\n')