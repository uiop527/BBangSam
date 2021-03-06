from __future__ import division

from tkinter import *
from tkinter import messagebox
import tkinter.font

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/facesamples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3face.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='face.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/face/face.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.2, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False)
classes = load_classes(opt.class_path) # Extracts class labels from file
print(classes)
print(opt.class_path)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)


    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
print ('\nSaving images:')

# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    print(path)
    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    #img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print(cls_pred)
            print(classes)
            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
        
                        
            
            #check bounding box and type casting
            int_tensor_x=x1.int()
            position_x = int_tensor_x.item()
            int_tensor_y=y1.int()
            position_y = int_tensor_y.item()
            int_tensor_width=box_w.int()
            value_w = int_tensor_width.item()
            int_tensor_height=box_h.int()
            value_h = int_tensor_height.item()
            
            """
            center_x = position_x + int(value_w/2)
            center_y = position_y + int(value_h/2)
            weight = int(value_w)
            height = int(value_h)
            
            #index 0~10 are hair's color, index 11~15 are skin's color
            positionarr = [[0 for col in range(16)] for row in range(2)]
            #select postion for hair color
            positionarr[0][0] = position_x                      #x???
            positionarr[1][0] = position_y + height             #y???
            positionarr[0][1] = position_x + weight/4
            positionarr[1][1] = position_y + height           
            positionarr[0][2] = center_x
            positionarr[1][2] = position_y + height
            positionarr[0][3] = position_x + 3*weight/4
            positionarr[1][3] = position_y + height
            positionarr[0][4] = position_x + weight
            positionarr[1][4] = position_y + height
            
            positionarr[0][5] = position_x                     #?????? hair point
            positionarr[1][5] = position_y + height - 20             
            positionarr[0][6] = position_x + 20
            positionarr[1][6] = position_y + height           
            positionarr[0][7] = position_y + 20
            positionarr[1][7] = position_y + height - 20
            positionarr[0][8] = position_x + weight
            positionarr[1][8] = position_y + height - 20
            positionarr[0][9] = position_x + weight - 20
            positionarr[1][9] = position_y + height - 20
            positionarr[0][10] = position_x + weight - 20
            positionarr[1][10] = position_y + height
            
            
            #select postion for skin color
            positionarr[0][11] = position_x + weight/4
            positionarr[1][11] = center_y
            positionarr[0][12] = position_x + 3*weight/4
            positionarr[1][12] = center_y
            positionarr[0][13] = center_x
            positionarr[1][13] = center_y
            positionarr[0][14] = center_x
            positionarr[1][14] = position_y + height/4
            positionarr[0][15] = center_x
            positionarr[1][15] = position_y + 3*height/4
            
            #color extract      
            #hair color extract
            r_sum = 0
            g_sum = 0
            b_sum = 0
            
            for i in range(0,11):
                r,g,b = (img[int(positionarr[1][i]), int(positionarr[0][i])])
                r_sum = r_sum + r
                g_sum = g_sum + g
                b_sum = b_sum + b
                hcr = r_sum/11
                hcg = g_sum/11
                hcb = b_sum/11
       
            #skin color extract    
            r_sum = 0
            g_sum = 0
            b_sum = 0
            
            for i in range(11,16):
                r,g,b = (img[int(positionarr[1][i]), int(positionarr[0][i])])
                r_sum = r_sum + r
                g_sum = g_sum + g
                b_sum = b_sum + b
                scr = r_sum/5
                scg = g_sum/5
                scb = b_sum/5
           
            pcolor = ""
            
            if scg <= 171 and scb <= 131:
                #skin warm
                if (hcr + hcg + hcb)/3 > 63:
                    print('spring')
                    pcolor = 'spring'
                else:
                    print('fall')
                    pcolor = 'fall'
            elif scg >= 187 and scb >= 164:
                #skin cold
                if (hcr + hcg + hcb)/3 > 63:
                    print('summer')
                    pcolor = 'summer'
                else:
                    print('winter')
                    pcolor = 'winter'
            else:
                if scb + scg < 327:
                    #skin warm
                    if (hcr + hcg + hcb)/3 > 63:
                        print('spring')
                        pcolor = 'spring'
                    else:
                        print('fall')
                        pcolor = 'fall'
                else:
                    #skin cold
                    if (hcr + hcg + hcb)/3 > 63:
                        print('summer')
                        pcolor = 'summer'
                    else:
                        print('winter')
                        pcolor = 'winter'
            """
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
            
    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
    """
root = Tk()
root.title("Color")
root.geometry("400x50")

font = tkinter.font.Font(family = "Arial", size = 20, slant="italic")
lbl = Label(root, text = "????????? ????????? " + pcolor + "?????????.", font = font)
lbl.pack()

    
root.mainloop()
    """