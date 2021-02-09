import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import torchvision.transforms as T
import errno
import time
import cv2
import os
import json
from collections import defaultdict
from collections import Counter
from PIL import Image

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

#getPrediction gets the bounding boxes and class of the FCNN predictions
def getPrediction(frameRGB, threshold):
    img = Image.fromarray(frameRGB)
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    predClass = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    predBoxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    predScore = list(pred[0]['scores'].detach().numpy())
    predT = [predScore.index(x) for x in predScore if x > threshold][-1] # Get list of index with score greater than threshold.
    predBoxes = predBoxes[:predT+1]
    predClass = predClass[:predT+1]
    return predBoxes, predClass

cvDataPath = 'torchData.json'

#CollectCVData will read in video fiels, use motion, face, and catface detection, compare them, store positional data and collect the detected areas as jpg for later model training
def CollectCVData():
    
    paths = ['/home/derek/securityCams/cmpVids/']
    #Load existing CVData files, load them into defaultdict, and skp existing file names
    cvData = defaultdict(None)
    with open(cvDataPath, 'r') as oldCVDataFile:
        cvData = defaultdict(None, json.load(oldCVDataFile))
        oldCVDataFile.close()
    for p in paths:
        
        # get a list of files
        fList = filter(lambda f: f.split('.')[-1] == 'mp4', os.listdir(p))
        # Loop through file names, and perform ranking if they do not already exist
        for fName in fList:
            if fName in cvData:
                print('file ' + fName + ' already exists in rankings.  SKIPPING')
                continue
            print('file: ', p+fName)
            cvData = Detect(p, fName, cvData)
            with open(cvDataPath, 'w') as outfile:
                json.dump(cvData, outfile)
                outfile.close()
#Detect will check each frame for motion, faces, and cats then log there location data to a json file and store detected areas as jpg
def Detect(p, fName, cvData):
    classCounter = Counter() #keep track of how many times a class is predicted for manimg purposes
    # Read the source video file
    vid = cv2.VideoCapture(p + fName)
    retry = 0
    while not vid.isOpened() and retry < 10:
        retry += 1
        print('vid not opened? retry: ', retry) 
        time.sleep(1)
        vid = cv2.VideoCapture(p + fName)

    motionDetected = False #if no motion gets detected skip wirtting video file
    frameWidth = int(vid.get(3)) #get width of origional frame
    frameHeight = int(vid.get(3)) #get height of origional frame

    frameReadFails = 0 #track how many failed frame reads there are TODO maybe 1 every video due to logical structure

    cvData[fName] = defaultdict(None) #cvData object for the video file
    data = cvData[fName] #object for this video file data
    frameTotal = 0 #counts total number of frames
    readSuccess = True #will control the video frame loops
    
    try:
        os.makedirs('torch/' + fName[:len(fName)-4] + '/' )
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    #new video file for bounding boxes
    mp4Vid = cv2.VideoWriter('torch/'+fName,cv2.VideoWriter_fourcc(*'mp4v'), 15, (frameWidth,frameHeight))
    
    #loop over video framse, check for motion, then faces etc,, then write bounding box data to cvData object, extract detected areas as jpg, and create new video with bounding boxes
    while readSuccess:
        #read video file 
        (readSuccess, frame) = vid.read()

        if not readSuccess:
            continue

        frameTotal += 1
             
        #set json objects for logging json data
        data[frameTotal] = defaultdict(None)
        frameData = data[frameTotal]
        
        ###new detection stuff
        rectTh = 3
        textTh = 3
        textSize = 3
        threshold = 0.5

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, predClass = getPrediction(frameRGB, threshold) 
        for i in range(len(boxes)):
            title = predClass[i]+str(classCounter[predClass[i]]) #class name + number of occurance for naming
            classCounter.update(predClass[i]) #update the class name counter

            cv2.rectangle(frameRGB, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rectTh)
            cv2.putText(frameRGB, title, boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, textSize, (0,255,0),thickness=textTh)
            (x, y) = boxes[i][0]
            (xw, yh) = boxes[i][1]
            predROI = frame[ y:yh, x:xw ]

            predROIFilePath = 'torch/' + fName[:len(fName)-4] + '/' + predClass[i] + '/' + title + '.jpg'
            frameData[title] = {'x': int(x),'y':int(y), 'w':int(xw - x), 'h':int(yh - y),'imgPath': predROIFilePath}

        #
        #MOTION REMOVED
        #

        cv2.imwrite('torch/' + fName[:len(fName)-4] +'/frame_'+ str(frameTotal) + '.jpg', frameRGB)
        #write frame to video writter
        mp4Vid.write(frameRGB)
    #End frame loop

    #Release video capture object
    vid.release() #release source video file
      
    mp4Vid
    mp4Vid.release()
    
    #remove empty ranking data and return rankings data
    if len(cvData[fName]) < 1:
        del cvData[fName]
    return cvData

if __name__ == "__main__":
    print('starting ranking')
    CollectCVData()
    print('finished ranking')
