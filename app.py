
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf
import sys
from pathlib import Path

import datetime
from ultralytics import YOLO 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from src import model
import torch
import torchvision.transforms.functional as tf
import sys
from pathlib import Path

import matplotlib.pyplot as plt



def main():
	parser = argparse.ArgumentParser(description="Process video with advertisement overlay using a model.")

	# Argument 1: Model file path
	parser.add_argument('--model', type=str,metavar="w",default="Models/cup_detector.pt" ,help="Path to the model file.")

	parser.add_argument('--smoothness', type=int,metavar="s",default=1 ,help="how smooth should the tracking boxes be")

	# Argument 2: Video file path
	parser.add_argument('--video', type=str,metavar="v",default="Input_videos/Podcast.mp4", help="Path to the input video file.")

	# Argument 3: Advertisement file path
	parser.add_argument('--advertisement', type=str,metavar="ad",default="Input_ads/nescafe_cup.png", help="Path to the advertisement file.")

	args = parser.parse_args()


	#IMPORTANT


	model = YOLO(args.model)
	source = args.video
	image_path = args.advertisement
	results = model.track(source = source,device = "cpu")
	output_path = 'Output_video/demo_cup.mp4'
	no_of_frames = 1000
	smooth = args.smoothness # The degree of smoothness. 

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



	video_path = source # can be found in the data link will be shared if you want to test out your own
	ad_image_path = image_path # Can be found in the data
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
	output_path = 'Output_video/demo_cup.mp4' # output video naame





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
	print('load Harmonizer...\n')
	harmonizer = model.Harmonizer()
	if cuda:
		harmonizer = harmonizer.cuda()
		harmonizer.load_state_dict(torch.load('./pretrained/harmonizer.pth'), strict=True)
	else:
		harmonizer.load_state_dict(torch.load('./pretrained/harmonizer.pth', map_location='cpu'), strict=True)
	harmonizer.eval()
	print('Harmonizer loaded...\n')


	# define example paths



	# read input videos
	bg_frames, fps = load_video_frames(bg_video_path)
	fps = 25
	ema = 1 - 1 / fps
	print('Background Video Loaded loaded...\n')
	image_data = cv2.imread(ad_image_path , cv2.IMREAD_UNCHANGED)
	print('Ad image loaded...\n')
	h, w = bg_frames[0].shape[:2]

	# define video writer

	h, w = bg_frames[0].shape[:2]
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	harmonized_vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

	# harmonization
	ema_arguments = None
	print("start harmonizing")
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
	
	
if __name__ == "__main__":
    main()
