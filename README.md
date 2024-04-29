to run this program please install
pip install ultralytics

1.	Model selection: YOLOv8
I am using YOLOv8 for high FPS detection and good accuracy model. YOLOV8 from Ultralytics is easy to train, and inference. The data i trained on is from BeyBlade video from YouTube. The labelling process is divide into two steps:
•	First: Label manual all extracted frames using LabelImg, and then train YOLOv8 model using labelled images.
•	Second: Use the trained model on new images to generate labels, review and correct the new labels using LabelImg. Repeat these steps until we achieve our desired quantity of data.
yolov8m.pt is used for transfer learning.

3.	Logic behind program:
 
To draw area, press s and click-drag mouse
 
To know any beyblade stopped, we detect any stop class from YOLOv8, and count how many stop class is appear in a frame, if stop class appear more than 10 frame then the match is over
 
To know if any beyblade is outside from arena, we check if center of bounding boxes from detected object is outside from Area
