from configClass import ConfigClass
import face_recognition
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import pickle
import torchvision.transforms as T
import errno
import time
import cv2
import os, sys, getopt
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
def knownFace(imgROI, knownFaceEncodings, knownFaceNames):
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
            return name
    return None
        

#getPrediction gets the bounding boxes and class of the FCNN predictions
def getPrediction(frameRGB, threshold):
    img = Image.fromarray(frameRGB)
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    predClass = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    predBoxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    predScore = list(pred[0]['scores'].detach().numpy())
    noMatch = True
    for x in predScore:
        if x > threshold:
            noMatch = False
    if noMatch:
        pass
    else: 
        predT = [predScore.index(x) for x in predScore if x > threshold][-1] # Get list of index with score greater than threshold.
        predBoxes = predBoxes[:predT+1]
        predClass = predClass[:predT+1]
        return predBoxes, predClass
    return [], []

#loadKnownFaces loads the know faces
def loadKnownFaces(config):
    f = open(config.knownFaceNamesPath,"rb")
    refDict = pickle.load(f)        
    f.close()

    f = open(config.knownFaceEncodingsPath,"rb")
    embedDictt = pickle.load(f)      
    f.close()
    knownFaceEncodings = []  
    knownFaceNames = []

    for refId , embedList in embedDictt.items():
        for myEmbed in embedList:
            knownFaceEncodings += [myEmbed]
            knownFaceNames += [refDict[refId]]

    return knownFaceEncodings, knownFaceNames

#recurvePath will follow paths recursivly
def recurvePath(path, jpgList):
    fList = os.listdir(path)
    for fName in fList:
        if os.path.isdir(path + fName):
            jpgList = recurvePath(path + fName + '/', jpgList)
        elif fName[len(fName)-4:].lower() == '.jpg' or fName[len(fName)-4:].lower() == 'png':
            jpgList.append({'path': path, 'fName': fName})
    return jpgList


#loadCVData will load or create and load the cvData file
def loadCVData(config):
    try:
        with open(config.cvDataPath, 'r') as oldCVDataFile:
            cvData = defaultdict(None, json.load(oldCVDataFile))
            oldCVDataFile.close()
            return cvData
    except FileNotFoundError:
        try:
            with open(config.cvDataPath, 'w') as f:
                f.write('{}')
                f.close()
            with open(config.cvDataPath, 'r') as oldCVDataFile:
                cvData = defaultdict(None, json.load(oldCVDataFile))
                oldCVDataFile.close()
                return cvData
        except:
            raise
    except:
        raise
    
#CollectCVData will read in video fiels, use motion, face, and catface detection, compare them, store positional data and collect the detected areas as jpg for later model training
def CollectCVData(config):
    #Load existing CVData files, load them into defaultdict, and skp existing file names
    cvData = defaultdict(None)
    #open or create cvData file
    cvData = loadCVData(config)

    kEncodings, kNames = loadKnownFaces(config)

    jpgList = []
    for p in config.imagePaths:
        jpgList = recurvePath(p, jpgList)
    print('len: ', len(jpgList))
    
    classCounter = Counter() #keep track of how many times a class is predicted for manimg purposes
    for jpg in jpgList:
        print('jpg: ',jpg)
        if (jpg['path'] + jpg['fName']) in cvData:
            continue
        cvData = Detect( jpg['path'], jpg['fName'], cvData,classCounter, config, kEncodings, kNames)
        with open(config.cvDataPath, 'w') as outfile:
            json.dump(cvData, outfile)
            outfile.close()
            
#Detect will check each frame for motion, faces, and cats then log there location data to a json file and store detected areas as jpg
def Detect(path, fName, cvData, classCounter, config, kEncodings, kNames):
    img = cv2.imread(path +fName)
    #resize image if desired
    if config.scalePercent != 100:
        width = int(img.shape[1] * config.scalePercent / 100)
        height = int(img.shape[0] * config.scalePercent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    cvData[path + fName] = defaultdict(None) #cvData object for the video file
    data = cvData[path + fName] #object for this video file data
    
    print(path + fName)
    try:
        os.makedirs(config.baseOutputPath )
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('not exist error?')
            raise
    
        start = time.time()
        
        try:
           imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print('some kind of error with the file')
            del cvData[path + fName]
            return cvData
        namedPeople = []
        boxes, predClass = getPrediction(imgRGB, config.drawConfig.threshold) 
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
                name = knownFace(predROI, kEncodings, kNames)
                if name != None and name != 'unknown':
                    predClass[i] = name
                    namedPeople.append(name)
            if classCounter[predClass[i]] == 0:
                try:
                    os.makedirs(config.baseOutputPath + predClass[i])
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            
            title = predClass[i]+str(classCounter[predClass[i]]) #class name + number of occurance for naming
            classCounter.update([predClass[i]]) #update the class name counter

            predROIFilePath = config.baseOutputPath + predClass[i] + '/' + title + '.jpg'
            cv2.imwrite(predROIFilePath, predROI)
            data[title] = {'x': x,'y':y, 'w':xw - x, 'h':yh - y,'imgPath': predROIFilePath}
            cv2.rectangle(imgRGB, (x, y), (xw, yh),color=(0, 255, 0), thickness=config.drawConfig.rectThickness)
            cv2.putText(imgRGB, title, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, config.drawConfig.textSize, (0,255,0),thickness=config.drawConfig.textThickness)
        named = ''
        for n in namedPeople:
            named = named+n
        #if names were found write image to named directory
        if len(namedPeople) > 0:
            try:
                os.makedirs(config.baseOutputPath + named)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            print('torch/images/' + fName)
            cv2.imwrite(config.baseOutputPath + named + '/' + fName, imgRGB)
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
    start = time.time()
    configPath = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hc:",["configFile="])
    except getopt.GetoptError:
        print('torchCVDataCollect.py -c <configFile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('torchCVDataCollect.py - <configFile>')
            sys.exit()
        elif opt in ("-c", "--configFile"):
            configPath = arg

    print('starting ranking')
    if configPath != '':
        config = ConfigClass(filePath=configPath)
    else:
        config = ConfigClass()

    CollectCVData(config)
    print('finished ranking in: ', (time.time() - start))
