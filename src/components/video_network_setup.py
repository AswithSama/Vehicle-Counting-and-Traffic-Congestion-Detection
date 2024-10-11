import cv2 as cv
import numpy as np
import sys
import os
# Assuming 'utils.py' is in a directory one level above your current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exception import CustomException
from logger import logging

def initialize_video_and_network(video_path, cfg_path, weights_path, coco_names_path):
    try:
        video = cv.VideoCapture(video_path)
        logging.info("Successfully loaded the video")
        video.set(cv.CAP_PROP_FPS, int(10))
        fps = video.get(cv.CAP_PROP_FPS)
        with open(coco_names_path) as f:
            labels = [line.strip() for line in f]
        network = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
        layersNames = network.getLayerNames()
        layersOutput = [layersNames[i - 1] for i in network.getUnconnectedOutLayers()]

        return video, network, layersOutput, labels
    except Exception as e:
        raise CustomException(e,sys)
