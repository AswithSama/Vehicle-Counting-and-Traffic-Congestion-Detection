import cv2 as cv
import numpy as np
import sys
import os
from exception import CustomException
# Assuming 'utils.py' is in a directory one level above your current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def yolo_object_detection(network, layersOutput, frame, prob_min, threshold):
    try:
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (416, 416))
        network.setInput(blob)
        output_from_network = network.forward(layersOutput)

        bounding_boxes = []
        confidences = []
        class_numbers = []

        for results in output_from_network:
            for detectedObjects in results:
                scores = detectedObjects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > prob_min:
                    box_current = detectedObjects[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv.dnn.NMSBoxes(bounding_boxes, confidences, prob_min, threshold)
        return results, bounding_boxes, class_numbers
    except Exception as e:
        raise CustomException(e,sys)
        