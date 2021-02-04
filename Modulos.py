import cv2
import numpy as np
import matplotlib.pyplot as plt

import skimage.io, skimage.feature

import os, shutil
import random

from PIL import Image

######################################################################################################

def Imprimir_Imagenes(Imagenes,n_rows=5,n_cols=5):
    """La siguiente función nos permite imprimir de manera ordenada una cierta cantidad de imagenes"""
    
    if n_rows != 1 and n_cols != 1:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,10))
    else:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5,15))
        if n_rows == 1:
            axes.resize((1,n_cols))
        else:
            axes.resize((n_rows,1))

    imgn = 0

    for i in range(len(axes)):

        for j in range(axes.shape[1]):

            axes[i,j].imshow(Imagenes[imgn],cmap="gray")
            axes[i,j].set_xticklabels([])
            axes[i,j].set_yticklabels([])
            axes[i,j].axis("off")

            imgn += 1

    plt.tight_layout()
    plt.show()
    
######################################################################################################

anchura, altura = (150,150) # Anchura y Altura del patch a ingresar a nuestra CNN para reconocimiento.

import datetime
import numpy as np
import cv2

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Extraemos el rostro de cada frame:
def extraccion_patches(frame, left, top, right, bottom):
    
    # Extraemos el/los rostro(s) de cada frame:
    
    if (left > 0 and right < frame.shape[1]): # En caso de que el patch sobre pase (por error) la anchura de la imagen.
        
        if (top > 0 and bottom < frame.shape[0]):
            
            patch = frame[top:bottom, left:right]
            
        else:
            
            patch = frame[0:frame.shape[0], left:right]
            
    else: # En caso de que el patch sobre pase (por error) la altura de la imagen original
        
        if (top > 0 and bottom < frame.shape[0]):
            
            patch = frame[top:bottom, 0:frame.shape[1]]
            
        else:
            
            patch = frame[0:frame.shape[0], 0:frame.shape[1]]
        
    patch = cv2.resize(patch,(anchura,altura)) # Reescalamos para poder entrenar nuestra red.
   
    return patch


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    
    # Extracting the patch where a face appears:
    patch = extraccion_patches(frame, left, top, right, bottom)
    
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)
    
    return patch, [(left,bottom), (right,top)]


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    rostros_por_frame = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    
    patches = list()
    coordenadas_lista = list()
    
    for i in rostros_por_frame: # for i in rostros_en_el_frame:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        patch, coordenadas = draw_predict(frame, confidences[i], left, top, right, bottom)
        patches.append(patch)
        coordenadas_lista.append(coordenadas)
        
    patches = np.asarray(patches)
    
    return final_boxes, patches, coordenadas_lista


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom
