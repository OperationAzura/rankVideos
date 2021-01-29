import cv2
import os
import json
from collections import defaultdict

#CompressVideos will read in all the video fiels, compress them using ffmpeg and rename them based on OCR'd image data
def CompressVideos():
    
    paths = ['/home/derek/securityCams/cmpVids/']
    
    for p in paths:
        print('path :', p)
        # get a list of files
        fList = filter(lambda f: f.split('.')[-1] == 'mp4', os.listdir(p))
        rankings = defaultdict(int)
        # encode each file
        for fName in fList:
            
            RankVids(p, fName, rankings)
        with open('rankings.json', 'w') as outfile:
            json.dump(rankings, outfile)

#RankVids will read in the video and check for pedestrians
def RankVids(p, fName, rankings):
    # Read the source video file
    vid = cv2.VideoCapture(p + fName)
    frameCnt = 0
    firstFrame = None
    #classifiers
    pedestrianClassifier = 'pedestrian.xml'

    #trackers
    pedestrianTracker = cv2.CascadeClassifier(pedestrianClassifier)

    readSuccess = True
    print('fName: ', fName)
    while readSuccess:
        #read video file
        (readSuccess, frame) = vid.read()

        if readSuccess:
            #convert to grey scale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #blur the frame for motion detection
            grayBlurFrame = cv2.GaussianBlur(grayFrame, (25, 25), 0)
            if frameCnt == 0:
                firstFrame = grayBlurFrame
            else if frameCnt > 20:
                frameCnt = 0
        else:
            break

        #motion detection comparisons
        deltaframe = cv2.absdiff(firstFrame,grayBlurFrame)
        threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold,None)
        countour, heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ##############
        
        # Detect Pedestrians
        pedestrians = pedestrianTracker.detectMultiScale(grayFrame,1.1,9)

        rankings[fName] += 1

        # Draw square around the pedestrians
        #for (x, y, w, h) in pedestrians:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #Release video capture object
    vid.release()

if __name__ == "__main__":
    print('starting ranking')
    CompressVideos()
    print('finished ranking')
