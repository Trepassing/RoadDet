import cv2
import os, glob
import argparse

parser = argparse.ArgumentParser(description='getVideo')
parser.add_argument('--input', default='runs/detect/exp', type=str, help='Input images')
parser.add_argument('--output', default='shiqu2.mp4', type=str, help='Directory for output')

args = parser.parse_args()
path = args.input
new_path = args.output

fps = 2    #保存视频的FPS，可以适当调整
dst = "video/" + new_path
print(dst)
fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
videoWriter = cv2.VideoWriter(dst,fourcc,fps,(960,540))#最后一个是保存图片的尺寸  1920,1080
imgs=glob.glob(path+'/*.jpg')
for imgname in imgs:
    frame = cv2.imread(imgname)
    frame = cv2.resize(frame,None,fx=0.5,fy=0.5)
    videoWriter.write(frame)
videoWriter.release()