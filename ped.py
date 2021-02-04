import time
import cv2
import os
import json
from collections import defaultdict

#RankVideos will read in all the video fiels, compress them using ffmpeg and rename them based on OCR'd image data
def RankVideos():
    
    paths = ['/home/derek/securityCams/cmpVids/']
    #Load existing ranking files, load them into defaultdict, and skp existing file names
    rankings = defaultdict(None)
    #with open('rankings.json', 'r') as oldRankFile:
    #    rankings = defaultdict(None, json.load(oldRankFile))
    for p in paths:
        print('path :', p)
        # get a list of files
        fList = filter(lambda f: f.split('.')[-1] == 'mp4', os.listdir(p))
        # Loop through file names, and perform ranking if they do not already exist
        for fName in fList:
            #if fName in rankings:
            #    print('file ' + fName + ' already exists in rankings.  SKIPPING')
            #    continue
            print(p+fName)
            rankings = RankVids(p, fName, rankings)
            if len(rankings) > 0 :
                print('rankings = ', rankings)
                #with open('newRankings.json', 'w') as outfile:
                #    json.dump(rankings, outfile)
                #    outfile.close()
#RankVids will check each frame for motion and then pedestrians and log there location data to a json file
def RankVids(p, fName, rankings):
    # Read the source video file
    rected = False
    vid = cv2.VideoCapture(p + fName)
    while not vid.isOpened():
        print('vid not opened?')
        time.sleep(1)
        vid = cv2.VideoCapture(p + fName)
    #(readSuccess, frame) = vid.read()
    #print(readSuccess)
    #(xx, yy, ww, hh) = cv2.boundingRect(frame)

    #print('going')
    frameCnt = 0 #count frames for motion detection count 
    firstFrame = None #frame to compare next frames to for motion detection
    rankings[fName] = defaultdict(None) #ranking object for file
    rank = rankings[fName] #object for this video file ranking
    frameTotal = 0 #counts total number of frames
    readSuccess = True #will control the video frame loops
    #classifier file names
    pedestrianClassifier = 'pedestrian.xml'
    #trackers
    pedestrianTracker = cv2.CascadeClassifier(pedestrianClassifier)
    
    #
    #for new vid stuff
    #
    w = 1080
    h = 720
    #boundedFrames = []
    #new video file for bounding boxes
    ### cv2.VideoWriter_fourcc(*'DIVX')
    mp4Vid = cv2.VideoWriter('mp4_'+fName,cv2.VideoWriter_fourcc(*'FMP4'), 15, (int(vid.get(3)),int(vid.get(4))))
    #x264Vid = cv2.VideoWriter('x264_'+fName,cv2.VideoWriter_fourcc(*'x264'), 15, (w,h))
    #h264Vid = cv2.VideoWriter('h264_'+fName,cv2.VideoWriter_fourcc(*'h264'), 15, (w,h))
    #hvecVid = cv2.VideoWriter('hvec_'+fName,cv2.VideoWriter_fourcc(*'hvec'), 15, (w,h))
    #x265Vid = cv2.VideoWriter('x265_'+fName,cv2.VideoWriter_fourcc(*'x265'), 15, (w,h))
    #h265Vid = cv2.VideoWriter('h265_'+fName,cv2.VideoWriter_fourcc(*'h265'), 15, (w,h))
    #boundedVid = cv2.VideoWriter('bounded_'+fName,cv2.VideoWriter_fourcc(*'avc1'), 15, (w,h))
    
    #print('fName: ', fName)
    #loop over video framse, check for motion, then check for pedestrians, then write bounding box data to rankings object
    while readSuccess:
        rected = False
        #read video file 
        (readSuccess, frame) = vid.read()

        if readSuccess:
            frameTotal += 1
            #
            #set new video frame
            # to be overwritten with bounded boxes
            #boundedFrames.append(frame)
            #
            #
            #
            colorFrame = frame #cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB)
            #colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_RGB2BGR)
            #convert frame to grey scale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #blur gray frame for motion detection
            grayBlurFrame = cv2.GaussianBlur(grayFrame, (25, 25), 0)
            #reset firstFrame ever 20 frames
            if frameCnt == 0:
                firstFrame = grayBlurFrame
            elif frameCnt > 20:
                frameCnt = 0
        else:#if frame read fails, skip this loop and finish
            print('no read success')
            continue
        #increment motion frame count
        frameCnt += 1
        #motion detection comparisons
        deltaframe = cv2.absdiff(firstFrame,grayBlurFrame)
        threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold,None)
        countour, heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #initialize motion bounding box variables
        mx = 0 #motion x variable
        my = 0 #motion y variable
        mw = 0 #motion w variable
        mh = 0 #motion h variable
        #loop through detected contours that exceed a certian size
        for i in countour:
            if cv2.contourArea(i) < 1500:
                continue
            #bounding box of the motion
            (mx, my, mw, mh) = cv2.boundingRect(i)
            #draw box on origional frame
            cv2.rectangle(colorFrame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            cv2.putText(colorFrame, 'Motion', (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            #region of interest where motion was detected
            #motionROIGray = grayFrame[my:my+mh, mx:mx+mw]
            
            #scan frame for pedestrians then add data to rankings
            pedestrians = pedestrianTracker.detectMultiScale(grayFrame,1.1,9)
            pCount = 1 # pedestrian count for logging data
            #loop over detected pedestrians and get bounding box data
            for (px, py, pw, ph) in pedestrians:
                rank[ str(frameTotal)+ '_'  + str(pCount)] = {"px":int(px),"py":int(py),"pw":int(pw),"ph":int(ph),"mx":int(mx),"my":int(my),"mw":int(mw),"mh":int(mh)}
                pCount += 1
                #draw pedestrian box on origional frame
                cv2.rectangle(colorFrame, (px, py), (px + pw, py + ph), (255, 255, 0), 2)
                cv2.putText(colorFrame, 'Human', (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                rected = True
            #End pedestrian loop
        ###End motion detection loop
        #write frame with bounded boxes to video writer object
        if readSuccess:
            #boundedVid.write(colorFrame)
            ########
            if rected:
                cv2.imwrite('img_' + str(frameTotal) + '_' + fName + '.jpg', colorFrame)
            mp4Vid.write(colorFrame)
            
            #x264Vid.write(colorFrame)
            #h264Vid.write(colorFrame)
            #hvecVid.write(colorFrame)
            #h265Vid.write(colorFrame)
            #x265Vid.write(colorFrame)
    #End frame loop
    #Release video capture object
    vid.release()
    #boundedVid.release()
    ####
    mp4Vid
    mp4Vid.release()
    print('XXXXXXXXXXXXXXXXXXXXXXXX')
    print('XXXXXXXXXXXXXXXXXXXXXXXX')
    print('done with file: ' + fName )
    print(str(frameTotal) + ' frames written')
    print('to: mp4_'+fName+'v')
    #x265Vid.release()
    #h265Vid.release()
    #x264Vid.release()
    #h264Vid.release()
    #hvecVid.release()
    #remove empty ranking data and return rankings data
    if len(rankings[fName]) < 1:
        del rankings[fName]
    return rankings

if __name__ == "__main__":
    print('starting ranking')
    RankVideos()
    print('finished ranking')
