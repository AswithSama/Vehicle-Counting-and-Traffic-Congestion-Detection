import math
import sys
import os
from exception import CustomException
# Assuming 'utils.py' is in a directory one level above your current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import logging
def track_objects(no_of_frames, center_points_prev_frame, center_points_cur_frame, tracking_objects, object_id):
    try:
        if no_of_frames <= 2:
            for pt in center_points_prev_frame:
                for pt2 in center_points_cur_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    if distance < 30:
                        tracking_objects[object_id] = pt
                        object_id += 1
        else:
            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()
            for object_id, pt2 in tracking_objects_copy.items():
                object_existence = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0])
                    if distance < 20:
                        tracking_objects[object_id] = pt
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        object_existence = True
                        continue
                if not object_existence:
                    tracking_objects.pop(object_id)
            for pt in center_points_cur_frame:
                tracking_objects[object_id] = pt
                object_id += 1
        return tracking_objects, object_id
    except Exception as e:
        logging.info('error occured in vehicle tracking part')
        raise CustomException(e,sys)

