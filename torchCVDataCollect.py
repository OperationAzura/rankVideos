import face_recognition
import torch
import torchvision
device = torch.device("cuda:0")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

import pickle
import torchvision.transforms as T
import errno
import time
import cv2
import os
import json
import numpy as np
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

#knownFace will check if a person has a known face
def knownFace(imgROI):
    faceLocations = face_recognition.face_locations(imgROI)
    faceEncodings = face_recognition.face_encodings(imgROI, faceLocations)
    faceNames = []

    for faceEncoding in faceEncodings:
        matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
        name = "Unknown"
        faceDistances = face_recognition.face_distance(knownFaceEncodings, faceEncoding)
        bestMatchIndex = np.argmin(faceDistances)

        if matches[bestMatchIndex]:
            name = knownFaceNames[bestMatchIndex]
            print('XXX')
            print('name: ', name)
            print('XXX')
            return name
    return None
        

#getPrediction gets the bounding boxes and class of the FCNN predictions
def getPrediction(frameRGB, threshold):
    img = Image.fromarray(frameRGB)
    #
    #move tensors to cuda?
    #
    #dtype = torch.float
    
    ts = T.ToTensor()
    print('ts: ',type(ts))
    transform = T.Compose([ts]) # Defing PyTorch Transform
    print('transform: ',type(transform))
    print('about to cuda the tensor')
    #transform.to(device='cuda')
    print('just cuda\'d the tensor')
    img = transform(img).to(device) #.to('cuda') # Apply the transform to the image
    
#########

    print('img: ', type(img))
    #model = model.to(device)
    
    #print(model.state_dict())
    pred = model([img]) # Pass the image to the model
    pred = pred.to(torch.device('cpu'))
    print(model.state_dict())
    predClass = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    predBoxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    predScore = list(pred[0]['scores'].detach().numpy())
    noMatch = True
    for x in predScore:
        if x > threshold:
            noMatch = False
    if noMatch:
        print("len(predScore) < 1 !!!!!!!!!!!!")
    else: 
        predT = [predScore.index(x) for x in predScore if x > threshold][-1] # Get list of index with score greater than threshold.
        predBoxes = predBoxes[:predT+1]
        predClass = predClass[:predT+1]
        return predBoxes, predClass
    return [], []

cvDataPath = 'torchData.json'
f = open("ref_name.pkl","rb")
refDict = pickle.load(f)        
f.close()

f = open("ref_embed.pkl","rb")
embedDictt = pickle.load(f)      
f.close()
knownFaceEncodings = []  
knownFaceNames = []

for refId , embedList in embedDictt.items():
    for myEmbed in embedList:
        knownFaceEncodings += [myEmbed]
        knownFaceNames += [refDict[refId]]

faceLocations = []
faceEncodings = []
faceNames = []


#recurvePath will follow paths recursivly
def recurvePath(path, jpgList):
    fList = os.listdir(path)
    for fName in fList:
        if os.path.isdir(path + fName):
            jpgList = recurvePath(path + fName + '/', jpgList)
        elif fName[len(fName)-4:].lower() == '.jpg' or fName[len(fName)-4:].lower() == 'png':
            jpgList.append({'path': path, 'fName': fName})
    return jpgList


#CollectCVData will read in video fiels, use motion, face, and catface detection, compare them, store positional data and collect the detected areas as jpg for later model training
def CollectCVData():
    
    paths = ['/cuda/projects/rankVideos/pictures/']

    #Load existing CVData files, load them into defaultdict, and skp existing file names
    cvData = defaultdict(None)
    with open(cvDataPath, 'r') as oldCVDataFile:
        cvData = defaultdict(None, json.load(oldCVDataFile))
        oldCVDataFile.close()
    jpgList = []
    for p in paths:
        jpgList = recurvePath(p, jpgList)
    print('len: ', len(jpgList))
    
    classCounter = Counter() #keep track of how many times a class is predicted for manimg purposes
    for jpg in jpgList:
        print('jpg: ',jpg)
        if (jpg['path'] + jpg['fName']) in cvData:
            
            continue
        cvData = Detect( jpg['path'], jpg['fName'], cvData,classCounter)
        with open(cvDataPath, 'w') as outfile:
            json.dump(cvData, outfile)
            outfile.close()
            
#Detect will check each frame for motion, faces, and cats then log there location data to a json file and store detected areas as jpg
def Detect(path, fName, cvData, classCounter):
    
    img = cv2.imread(path +fName)
    
    cvData[path + fName] = defaultdict(None) #cvData object for the video file
    data = cvData[path + fName] #object for this video file data
    
    print('before crash')
    print(path + fName)
    try:
        os.makedirs('torch/images/' )
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('not exist error?')
            raise
    
        start = time.time()
        
        ###new detection stuff
        rectTh = 3
        textTh = 3
        textSize = 3
        threshold = 0.5
        print('right before imgRGB')
        #print('len(img): ', len(img))
        try:
           imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print('some kind of error with the file')
            del cvData[path + fName]
            return cvData
        namedPeople = []
        boxes, predClass = getPrediction(imgRGB, threshold) 
        for i in range(len(boxes)):
            (x, y) = boxes[i][0]
            (xw, yh) = boxes[i][1]
            ### convert type numpy.float32 to int
            x = int(x)
            y = int(y)
            xw = int(xw)
            yh = int(yh)

            #crop predicted region of interest
            predROI = img[ y:yh, x:xw ]
            #check if a pseron was found, then check if they can be identified
            if predClass[i] == 'person':
                name = knownFace(predROI)
                if name != None and name != 'unknown':
                    predClass[i] = name
                    namedPeople.append(name)
            if classCounter[predClass[i]] == 0:
                try:
                    os.makedirs('torch/images/' + predClass[i])
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            
            title = predClass[i]+str(classCounter[predClass[i]]) #class name + number of occurance for naming
            classCounter.update([predClass[i]]) #update the class name counter

            predROIFilePath = 'torch/images/' + predClass[i] + '/' + title + '.jpg'
            cv2.imwrite(predROIFilePath, predROI)
            data[title] = {'x': x,'y':y, 'w':xw - x, 'h':yh - y,'imgPath': predROIFilePath}
            cv2.rectangle(imgRGB, (x, y), (xw, yh),color=(0, 255, 0), thickness=rectTh)
            cv2.putText(imgRGB, title, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, textSize, (0,255,0),thickness=textTh)
        print('right here?')
        named = ''
        for n in namedPeople:
            named = named+n
        #if names were found write image to named directory
        if len(namedPeople) > 0:
            try:
                os.makedirs('torch/images/' + named)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            print('torch/images/' + fName)
            cv2.imwrite('torch/images/' + named + '/' + fName, imgRGB)
        else:
            print('torch/images/' + fName)
            cv2.imwrite('torch/images/' + fName, imgRGB)
        print('image: ' + path + fName  )
        print(' processed in: ' + str(time.time() - start) + ' seconds')
        
    #remove empty ranking data and return rankings data
    if len(cvData[path + fName]) < 1:
        del cvData[path + fName]
    return cvData

if __name__ == "__main__":
    print('starting ranking')
    CollectCVData()
    print('finished ranking')
