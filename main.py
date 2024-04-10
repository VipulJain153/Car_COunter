from ultralytics import YOLO
import cv2,cvzone,math
from sort import *

model = YOLO("../YOLO-Weights/yolov8l.pt")

cap = cv2.VideoCapture("../Videos/cars.mp4") # For Videos
cap.set(3,1280)
cap.set(4,720)
cap.set(cv2.CAP_PROP_FPS,80)
cap.set(10,150)

FPS=cvzone.FPS()

# classNames = cvzone.classNames #coco dataset labels or added to cvzone manually by the below list
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

mask = cv2.imread("mask.png")

limits = [400,297,673,297]


total_count=[]
car = cv2.imread("car.png",cv2.IMREAD_UNCHANGED)
while True:
    detections=np.empty((0, 5))
    _,img = cap.read()
    img = cvzone.overlayPNG(img,car)
    imgRegion = cv2.bitwise_and(img,mask)
    # stream = True runs a generator for real time detection is is generally recommened to use
    results = model(imgRegion,stream=True)
    for j,i in enumerate(results):
        
        # print(i)
        boxes=i.boxes
        for box in boxes:
            # Bounding Box
            # print(box.xyxy,box.xywh) (xyxy is x1,y1-x2,x2) (xywh is x,y,w,h)
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            # x,y,w,h = box.xywh[0] 
            # bbox=int(x),int(y),int(w),int(h)
            bbox=x1,y1,x2-x1,y2-y1
            # className 
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100
            confStr = "{} {}".format(classNames[cls],conf) # Showing the Confidence

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus"\
                or currentClass == "motorbike" and conf>0.85:
                    # cvzone.cornerRect(img,bbox,l=9,rt=2)
                    # cvzone.putTextRect(img,confStr,(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                    if currentClass == "car":
                        currentArray = np.array([x1,y1,x2,y2,conf])
                        detections = np.vstack((detections,currentArray))

        resultsTracker = tracker.update(detections)
        cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

        for result in resultsTracker:
             x1,y1,x2,y2,Id = result
             x1,y1,x2,y2,Id = int(x1),int(y1),int(x2),int(y2),int(Id)
             w,h=x2-x1,y2-y1
             bbox=x1,y1,x2-x1,y2-y1
            #  print(result)
             cvzone.cornerRect(img,bbox,l=9,rt=2,colorR=(255,0,255))
             cvzone.putTextRect(img,f"{Id}",(max(0,x1),max(35,y1)),scale=2,thickness=3,offset=10)
             cx,cy = x1+w//2,y1+h//2
             cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

             if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
                  if total_count.count(Id)==0:
                    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
                    total_count.append(Id)
            
        cv2.putText(img,f"{len(total_count)}", (205,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),8)

    FPS.update(img,color=(0,255,0),pos=(25,720-50))
    cv2.imshow("Car Counter",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break