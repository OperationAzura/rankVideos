import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import pika
import torchvision.transforms as T
import errno
import time
import cv2
import os
import json
from collections import defaultdict
from collections import Counter
from PIL import Image
import base64
import sys

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
class VideoManager(object):
    def __init__(self, path, fileName, cvData):
        self.path = path
        self.fileName = fileName
        self.cvData = cvData
        self.classCounter = Counter() #keep track of how many times a class is predicted for manimg purposes
        self.vid = self.getVid()
        self.frameWidth = int(self.vid.get(3)) #get width of origional frame
        self.frameHeight = int(self.vid.get(4)) #get height of origional frame
        self.frameReadFails = 0
        self.cvData[self.fileName] = defaultdict(None) #cvData object for the video file
        self.data = self.cvData[self.fileName]
        self.frameTotal = 0 #counts total number of frames
        self.readSuccess = True #will control the video frame loops
        self.checkTorchDir()
        self.mp4Vid = cv2.VideoWriter('torch/'+self.fileName,cv2.VideoWriter_fourcc(*'mp4v'), 15, (self.frameWidth,self.frameHeight))

        #frameLoop is the main loop through the video frames
        def frameLoop(self):
            #loop over video framse, check for motion, then faces etc,, then write bounding box data to cvData object, extract detected areas as jpg, and create new video with bounding boxes
            while self.readSuccess:
                start = time.time()
                #read video file 
                (self.readSuccess, selfframe) = self.vid.read()
                if not readSuccess:
                    continue

                self.frameTotal += 1


        #getVid gets the video file based on the passed in file path
        def getVid(self):
            vid = cv2.VideoCapture(self.path + self.fileName)
            retry = 0
            while not vid.isOpened() and retry < 10:
                retry += 1
                print('vid not opened? retry: ', retry) 
                time.sleep(1)
                vid  = cv2.VideoCapture(self.path + self.fileName)
            return vid
        
        #checkTorchDir makes sure the directory exists or creates it
        def checkTorchDir(self):
            try:
                os.makedirs('torch/' + fName[:len(fName)-4] + '/' )
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        
        


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
    
    
        ###
        # ###
        # ###
        # ###
        # ###
        # ###
        ###      
        #set json objects for logging json data
        data[frameTotal] = defaultdict(None)
        frameData = data[frameTotal]
        
        ###new detection stuff
        rectTh = 3
        textTh = 3
        textSize = 3
        threshold = 0.5

        #
        #convert frame to bytes
        #send fram to MQ
        #

        print('frame type: ', type(frame))
        _,frameJPG = cv2.imencode('.jpg', frame)
        print('frameJPG type: ', type(frameJPG))
        jstr = json.dumps({"frame": base64.b64encode(frameJPG).decode('ascii'), 'fileName': fName, 'frameID': frameTotal})
        print('jstr type: ', type(jstr))

        connection = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@192.168.1.12:5672/'))
        channel = connection.channel()
        channel.queue_declare(queue='frame')
        channel.basic_publish(exchange='',
                      routing_key='frame',
                      body=jstr)
                      
        connection.close()

        #
        #Connect to response queue
        #
        con = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@localhost:5672/'))
        chan = con.channel()

        chan.queue_declare(queue='predictionResults')

        def callback(ch, method, properties, body):
            start = time.time()
            print('body: ',type(body))
            jBody = json.loads(body)
            for x in jBody:
                print('x: ',jBody[x])
        #
        # # consume 1?
        #     
        chan.basic_consume(queue='predictedResults', on_message_callback=callback, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        chan.start_consuming()
        sys.exit(0)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, predClass = getPrediction(frameRGB, threshold) 
        for i in range(len(boxes)):
            if classCounter[predClass[i]] == 0:
                try:
                    os.makedirs('torch/' + fName[:len(fName)-4] + '/' + predClass[i])
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            
            title = predClass[i]+str(classCounter[predClass[i]]) #class name + number of occurance for naming
            classCounter.update([predClass[i]]) #update the class name counter

            (x, y) = boxes[i][0]
            (xw, yh) = boxes[i][1]
            ### convert type numpy.float32 to int
            x = int(x)
            y = int(y)
            xw = int(xw)
            yh = int(yh)

            cv2.rectangle(frameRGB, (x, y), (xw, yh),color=(0, 255, 0), thickness=rectTh)
            cv2.putText(frameRGB, title, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, textSize, (0,255,0),thickness=textTh)
            #crop predicted region of interest
            predROI = frame[ y:yh, x:xw ]
            #save ROI image and data
            predROIFilePath = 'torch/' + fName[:len(fName)-4] + '/' + predClass[i] + '/' + title + '.jpg'
            cv2.imwrite(predROIFilePath, predROI)
            frameData[title] = {'x': x,'y':y, 'w':xw - x, 'h':yh - y,'imgPath': predROIFilePath}

        cv2.imwrite('torch/' + fName[:len(fName)-4] +'/frame_'+ str(frameTotal) + '.jpg', frameRGB)
        #write frame to video writter
        mp4Vid.write(frameRGB)
        #fmp4Vid.write(frameRGB)
        #frameBGR = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2BGR)
        #mp4BGRVid.write(frameBGR)
        #fmp4BGRVid.write(frameBGR)
        print('finished frame: ' + str(frameTotal) + 'of file: ' + fName)
        print(str(start - time.time()))
    #End frame loop

    #Release video capture object
    vid.release() #release source video file
      
    mp4Vid
    mp4Vid.release()
    #fmp4Vid
    #fmp4Vid.release()
    #mp4BGRVid
    #mp4BGRVid.release()
    #fmp4BGRVid
    #fmp4BGRVid.release()
    
    #remove empty ranking data and return rankings data
    if len(cvData[fName]) < 1:
        del cvData[fName]
    return cvData

if __name__ == "__main__":
    print('starting ranking')
    CollectCVData()
    print('finished ranking')
