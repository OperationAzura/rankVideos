import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import logging
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
        logging.basicConfig(filename='vidManager.log', encoding='utf-8', level=logging.DEBUG)

        self.maxActiveFrames = 4

        self.sentFrameCount = 0
        self.frameArr = []
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
        self.rectTh = 3
        self.textTh = 3
        self.textSize = 3
        self.threshold = 0.5
        logging.info('done with constructor')

    #frameLoop is the main loop through the video frames
    def frameLoop(self):
        #loop over video framse, check for motion, then faces etc,, then write bounding box data to cvData object, extract detected areas as jpg, and create new video with bounding boxes
        logging.info('starting frameLoop')
        while self.readSuccess:
            if self.sentFrameCount >= self.maxActiveFrames:
                logging.info('max active frames hit')
                time.sleep(10)
                continue
            
            while self. sentFrameCount > self.maxActiveFrames:
                print("maximum active frames hit! " + self.maxActiveFrames)
                time.sleep(1)
                
            start = time.time()
            #read video file 
            (self.readSuccess, self.frame) = self.vid.read()
            if not readSuccess:
                continue
            self.frameTotal += 1
            self.data[self.frameTotal] = defaultdict(None)
            self.frameData = self.data[self.frameTotal]
            _,self.frameJPG = cv2.imencode('.jpg', self.frame)
            self.jstr = json.dumps({"frame": base64.b64encode(self.frameJPG).decode('ascii'), 'fileName': self.fileName, 'frameID': self.frameTotal})
            self.sendFrame(self.jstr)
            self.frameArr.append({'frame': self.frame, 'data': self.frameData })
            self.sentFrameCount += 1
        self.finishVideo()
        logging.info('done with frameLoop')
        return self.cvData

    #finishVideo will finish the video once all the frames are collacted
    def finishVideo(self):
        logging.info('starting finishVideo')
        for i, frame in enumerate(self.frameArr):
            cv2.imwrite('torch/' + self.fileName[:len(self.fileName)-4] +'/frame_'+ i+1 + '.jpg', frame['frame'])
            self.mp4Vid.write(frame['frame'])
        self.mp4Vid
        self.mp4Vid.close()
        self.vid.close()
        logging.info('done with finishVideo')

    #predictionHandler gets the prediction boxes and clases from the MQ
    def predictionHandler(self):
        logging.info('starting predictionHandler')
        self.resCon = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@localhost:5672/'))
        self.resChan = self.resCon.channel()

        self.resChan.queue_declare(queue='predictionResults')

        def callback(ch, method, properties, body):
            start = time.time()
            print('body: ',type(body))
            jBody = json.loads(body)
            ###
            ### TODO use frameID to add boxes and classes to the frameArr
            ###
            ###
            self.frameArr[int(jBody['frameID'])-1]['boxes'] = jBody['boxes']
            self.frameArr[int(jBody['frameID'])-1]['classes'] = jBody['classes']

            boxes = jBody['boxes']
            classes = jbody['classes']
            for i in range(len(boxes)):
                if self.classCounter[classes[i]] == 0:
                    try:
                        os.makedirs('torch/' + self.fileName[:len(self.fileName)-4] + '/' + classes[i])
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                
                title = classes[i]+str(self.classCounter[classes[i]]) #class name + number of occurance for naming
                self.classCounter.update([classes[i]]) #update the class name counter

                (x, y) = boxes[i][0]
                (xw, yh) = boxes[i][1]
                ### convert type numpy.float32 to int
                x = int(x)
                y = int(y)
                xw = int(xw)
                yh = int(yh)

                cv2.rectangle(self.frameArr[int(jBody['frameID'])-1]['frame'], (x, y), (xw, yh),color=(0, 255, 0), thickness=self.rectTh)
                cv2.putText(self.frameArr[int(jBody['frameID'])-1]['frame'], title, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, self.textSize, (0,255,0),thickness=self.textTh)
                #crop predicted region of interest
                predROI = self.frameArr[int(jBody['frameID'])-1]['frame'][ y:yh, x:xw ]
                #save ROI image and data
                predROIFilePath = 'torch/' + self.fileName[:len(self.fileName)-4] + '/' + classes[i] + '/' + title + '.jpg'
                cv2.imwrite(predROIFilePath, predROI)
                self.frameData[title] = {'x': x,'y':y, 'w':xw - x, 'h':yh - y,'imgPath': predROIFilePath}

                for x in jBody:
                    print('x: ',jBody[x])
            #
            # # consume 
            #     
            self.resChan.basic_consume(queue='predictedResults', on_message_callback=callback, auto_ack=True)
            self.resChan.start_consuming()
            self.sentFrameCount -= 1
            logging.info('done with predictionHandler')
        

    #sendFrame sends the frame data to the MQ
    def sendFrame(self, msgString):
        logging.info('sending frame')
        self.sendMQ('frame', msgString)
        logging.info('frame sent')

    #sendMQ sends a message to a queue
    def sendMQ(self, queue, message):
        self.connection = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@192.168.1.12:5672/'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue)
        self.channel.basic_publish(exchange='',
                      routing_key=queue,
                      body=message)
                      
        self.connection.close()
    
    #getVid gets the video file based on the passed in file path
    def getVid(self):
        logging.info('getting video file: ' + self.fileName)
        vid = cv2.VideoCapture(self.path + self.fileName)
        retry = 0
        while not vid.isOpened() and retry < 10:
            retry += 1
            print('vid not opened? retry: ', retry) 
            time.sleep(1)
            vid  = cv2.VideoCapture(self.path + self.fileName)
        logging.info('got video file')
        return vid
    
    #checkTorchDir makes sure the directory exists or creates it
    def checkTorchDir(self):
        try:
            os.makedirs('torch/' + fName[:len(fName)-4] + '/' )
        except OSError as e:
            if e.errno != errno.EEXIST:

   
#CollectCVData will read in video fiels, use motion, face, and catface detection, compare them, store positional data and collect the detected areas as jpg for later model training
def CollectCVData():
    cvDataPath = 'torchData.json'
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
            vidManager = VideoManager(p, fName, cvData)
            cvData = vidManager.frameLoop()
            with open(cvDataPath, 'w') as outfile:
                json.dump(cvData, outfile)
                outfile.close()


if __name__ == "__main__":
    print('starting ranking')
    CollectCVData()
    print('finished ranking')
