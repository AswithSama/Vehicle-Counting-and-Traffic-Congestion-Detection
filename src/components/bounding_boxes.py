import cv2 as cv
import sys
import os
from exception import CustomException
# Assuming 'utils.py' is in a directory one level above your current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def draw_bounding_boxes(frame, results, bounding_boxes, class_numbers, labels, colours, center_points_cur_frame, for_counting_unique_vehicles):
    try:
        if len(results)==0:
            return None
        for i in results.flatten():
            if int(class_numbers[i]) in [2, 3, 5, 7]:  # Filtering specific vehicle classes
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                x_center = x_min + int(box_width / 2)
                y_center = y_min + int(box_height / 2)
                color_box_current = colours[class_numbers[i]].tolist()
                cv.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), color_box_current, 2)
                cv.circle(frame, (x_center, y_center), 4, (0, 0, 255), -1)
                center_points_cur_frame.append((x_center, y_center))
                for_counting_unique_vehicles.append(((x_center, y_center), int(class_numbers[i])))
                text_box_current = '{}'.format(labels[int(class_numbers[i])])
                cv.putText(frame, text_box_current, (x_min - 5, y_min), cv.FONT_HERSHEY_COMPLEX, 0.7, color_box_current, 2)
        return center_points_cur_frame
    except:
        raise CustomException(e,sys)
