import cv2 as cv
import numpy as np
from video_network_setup import initialize_video_and_network
from vehicle_detection import yolo_object_detection
from vehicle_tracking import track_objects
from vehicle_counting import count_vehicles
from bounding_boxes import draw_bounding_boxes
from exception import CustomException
import sys
import os

# Assuming 'utils.py' is in a directory one level above your current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#data_path='src/components/Data'
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to coco.names
file_path = os.path.join(current_dir, '..', 'Data')

from utils import get_video_path

def main():
    try:
        video_path = get_video_path()
        if video_path:
            weights_path = os.path.join(file_path, 'yolov3.weights')
            cfg_path = os.path.join(file_path, 'yolov3.cfg')
            coco_path = os.path.join(file_path, 'coco.names')
            video, network, layersOutput, labels = initialize_video_and_network(video_path, cfg_path, weights_path, coco_path)
        else:
            print("No file selected!")
            return
        
        # Variables initialization
        prob_min = 0.5
        threshold = 0.3
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        center_points_prev_frame = []
        tracking_objects = {}
        object_id = car_count = bike_count = bus_count = truck_count = 0
        count_list = []
        no_of_frames = 0
        
        while True:
            no_of_frames += 1
            center_points_cur_frame = []
            ret, frame = video.read()
            if not ret:
                break
            
            results, bounding_boxes, class_numbers = yolo_object_detection(network, layersOutput, frame, prob_min, threshold)
            center_points_cur_frame = draw_bounding_boxes(frame, results, bounding_boxes, class_numbers, labels, colours, center_points_cur_frame, for_counting_unique_vehicles=[])
            if center_points_cur_frame==None:
                continue
            tracking_objects, object_id = track_objects(no_of_frames, center_points_prev_frame, center_points_cur_frame, tracking_objects, object_id)
            counting_number_of_vehicles, car_count, bike_count, bus_count, truck_count = count_vehicles(frame, tracking_objects, count_list, counting_number_of_vehicles=0, for_counting_unique_vehicles=[], y_coordinate=frame.shape[0] - 200, car_count=0, bike_count=0, bus_count=0, truck_count=0)
            
            blank=np.zeros((500,500,3),dtype='uint8')
            cv.namedWindow('Count',cv.WINDOW_NORMAL)
            cv.putText(blank,'Number Of Cars: '+str(car_count),(30,30),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
            cv.putText(blank,'Number Of Bikes: '+str(bike_count),(30,70),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
            cv.putText(blank,'Number Of Buses: '+str(bus_count),(30,110),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
            cv.putText(blank,'Number Of Trucks: '+str(truck_count),(30,140),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
            cv.namedWindow('count',cv.WINDOW_NORMAL)
            cv.imshow('count',blank)
            cv.namedWindow('frame',cv.WINDOW_NORMAL)                      
            cv.imshow('frame',frame)
            center_points_prev_frame = center_points_cur_frame.copy()
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.waitKey(0)
    except Exception as e:
        raise CustomException(e,sys)

        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
