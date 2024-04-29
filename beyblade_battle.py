import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import csv
import imutils

cap = cv2.VideoCapture('video\\beyblade.mp4')
initBB = None
model = YOLO('runs\\detect\\train23\\weights\\best.pt')

def is_inside(inner_box, outer_box):
    inner_left, inner_top, inner_right, inner_bottom = inner_box
    outer_left, outer_top, outer_right, outer_bottom = outer_box

    # Calculate center coordinates of inner box
    inner_center_x = (inner_left + inner_right) / 2
    inner_center_y = (inner_top + inner_bottom) / 2

    # Check if the center of the inner box is inside the outer box
    return (
        outer_left <= inner_center_x <= outer_right and
        outer_top <= inner_center_y <= outer_bottom
    )
    
def check_stop_beyblade(count_stop_beyblade, bbboxes):
    beyblade1 = bbboxes[0]
    beyblade2 = bbboxes[1]
    class_beyblade1 = beyblade1.cls
    class_beyblade2 = beyblade2.cls
    if count_stop_beyblade > 10:
        if class_beyblade1 == 0:
            bb = beyblade1.xyxy[0]
            res = ['winner', (int(bb[0]),int(bb[1]), int(bb[2]),int(bb[3]))]
        else:
            bb = beyblade2.xyxy[0]
            res = ['winner', (int(bb[0]),int(bb[1]), int(bb[2]),int(bb[3]))]
    else:
        if class_beyblade1==1 or class_beyblade2==1:
            num = count_stop_beyblade+1
        else:
            num=0
        res= ['continue', num]
    return res
    
def check_outside_beyblade(area, bbboxes):
    beyblade1 = bbboxes[0]
    beyblade2 = bbboxes[1]
    if is_inside(beyblade1.xyxy[0], area) and is_inside(beyblade2.xyxy[0], area):
        return ['continue', 0]
    else:
        if is_inside(beyblade1.xyxy[0], area)==is_inside(beyblade2.xyxy[0], area):
            return ['draw', 0]
        if is_inside(beyblade1.xyxy[0], area):
            bb = beyblade1.xyxy[0]
            return ['winner', (int(bb[0]),int(bb[1]), int(bb[2]),int(bb[3]))]
        else:
            bb = beyblade2.xyxy[0]
            return ['winner', (int(bb[0]),int(bb[1]), int(bb[2]),int(bb[3]))]
    
def write_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['time', 'winner']) 
        writer.writerow(data)
        
count_stop_beyblade = 0
run = True
init_time=False
while run:
    ret, frame = cap.read()
    if ret==False:
        break
    frame = imutils.resize(frame, width=500)
    img_height, img_width, _ = frame.shape
    if initBB != None and initBB != (0,0,0,0):
        if init_time:
            print("init time")
            start_time=time.time()
            init_time=False
        predictions = model(frame, stream=True, verbose=False)
        for r in predictions:
            annotator = Annotator(frame)
            boxes = r.boxes
            # print(len(boxes))
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                bb = (int(initBB[0]), int(initBB[1]), int(initBB[0]+initBB[2]), int(initBB[1]+initBB[3]))
                annotator.box_label(b, model.names[int(c)])
                res = check_stop_beyblade(count_stop_beyblade, boxes)
                if res[0] != 'winner':
                    count_stop_beyblade=res[1]
                if res[0] == 'winner':
                    total_time = time.time() - start_time
                    data =[total_time, res[1]]
                    write_to_csv(data, 'output.csv')
                    cv2.waitKey(0)
                    run = False
                
                
                res = check_outside_beyblade(bb, boxes)
                if res[0] == 'winner':
                    total_time = time.time() - start_time
                    data =[total_time, res[1]]
                    write_to_csv(data, 'output.csv')
                    cv2.waitKey(0)
                    run = False

        frame = annotator.result()                     
        # results = model.track(frame, persist=True, tracker='bytetrack.yaml')
        # frame = results[0].plot()
        cv2.rectangle(
            frame, 
            (int(initBB[0]), int(initBB[1])), (int(initBB[0]+initBB[2]), int(initBB[1]+initBB[3])), (0, 255, 0), 2
        )
    if init_time==False:
        key = cv2.waitKey(1) & 0xFF
    else:
        key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
        init_time=True
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Frame', frame)
    
cap.release()
cv2.destroyAllWindows()
