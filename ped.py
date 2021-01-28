import cv2
import os
import json
from collections import defaultdict 

#CompressVideos will read in all the video fiels, compress them using ffmpeg and rename them based on OCR'd image data
def CompressVideos():
    global count
    paths = ['/home/derek/securityCams/cmpVids/']
    
    for p in paths:
        print('path :', p)
        # get a list of files
        fList = filter(lambda f: f.split('.')[-1] == 'mp4', os.listdir(p))
        rankings = defaultDict()
        # encode each file
        for fName in fList:
            count = count + 1
            RankVids(p, fName, rankings)
        with open('rankings.json', 'w') as outfile:
            json.dump(rankings, outfile)

#RankVids will read in the video and check for pedestrians
def RankVids(p, fName, rankings):
    # Read the source video file
    vid = cv2.VideoCapture(fName)

    #classifiers
    pedestrianClassifier = 'pedestrian.xml'

    #trackers
    pedestrianTracker = cv2.CascadeClassifier(pedestrianClassifier)

    readSuccess = True
    while readSuccess:
        #read video file
        (readSuccess, frame) = vid.read()

        if readSuccess:
            #convert to grey scale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # Detect Pedestrians
        pedestrians = pedestrian_tracker.detectMultiScale(gray_frame,1.1,9)

        rankings[fName] += 1

        # Draw square around the pedestrians
        #for (x, y, w, h) in pedestrians:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #Release video capture object
    vid.release()

