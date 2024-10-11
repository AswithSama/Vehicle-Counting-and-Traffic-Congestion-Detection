import cv2 as cv
import sys
import os
from exception import CustomException
# Assuming 'utils.py' is in a directory one level above your current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def count_vehicles(frame, tracking_objects, count_list, counting_number_of_vehicles, for_counting_unique_vehicles, y_coordinate, car_count, bike_count, bus_count, truck_count):
    try:
        cv.line(frame,(0,frame.shape[0]-200),(frame.shape[1],frame.shape[0]-200),(0,0,255),2)
        y_coordinate=frame.shape[0]-200
        for object_id, pt in tracking_objects.items():
            distance2 = pt[1] - y_coordinate
            if -10 < distance2 < 10:
                if object_id not in count_list:
                    counting_number_of_vehicles += 1
                    for i in for_counting_unique_vehicles:
                        if pt == i[0]:
                            if i[1] == 3:
                                bike_count += 1
                            elif i[1] == 2:
                                car_count += 1
                            elif i[1] == 5:
                                bus_count += 1
                            else:
                                truck_count += 1
                    count_list.append(object_id)
                cv.line(frame, (0, y_coordinate), (frame.shape[1], y_coordinate), (0, 255, 0), 2)
        return counting_number_of_vehicles, car_count, bike_count, bus_count, truck_count
    except Exception as e:
        raise CustomException(e,sys)
