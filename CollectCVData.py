import errno
import time
import cv2
import os
import json
from collections import defaultdict

cvDataPath = 'cvData.json'

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

    motionFrameLimit = 10 #Number of frames before resetting motionCompareFrame 
    motionFrameCount = 0 #count frames for motion detection count 
    motionCompareFrame = None #frame to compare next frames to for motion detection
    
    cvData[fName] = defaultdict(None) #cvData object for the video file
    data = cvData[fName] #object for this video file data
    frameTotal = 0 #counts total number of frames
    readSuccess = True #will control the video frame loops
    
    #classifier file names
    faceFrontPath = 'haarFaceFront.xml'
    faceProfilePath = 'haarFaceProfile.xml'
    eyePath = 'haarEye.xml'
    catFaceFrontPath = 'haarCatFaceFront.xml'
    
    #classifier objects
    faceFrontClassifier = cv2.CascadeClassifier(faceFrontPath)
    faceProfileClassifier = cv2.CascadeClassifier(faceProfilePath)
    eyeClassifier = cv2.CascadeClassifier(eyePath)
    catFaceFrontClassifier = cv2.CascadeClassifier(catFaceFrontPath)
    
    try:
        os.makedirs(fName + '/' )
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    #new video file for bounding boxes
    mp4Vid = cv2.VideoWriter(fName+'/'+fName,cv2.VideoWriter_fourcc(*'FMP4'), 15, (frameWidth,frameHeight))
    
    #loop over video framse, check for motion, then faces etc,, then write bounding box data to cvData object, extract detected areas as jpg, and create new video with bounding boxes
    while readSuccess:
        #read video file 
        (readSuccess, frame) = vid.read()

        if readSuccess:
            frameTotal += 1
            
            #TODO maybe not needed
            colorFrame = frame
            #convert frame to grey scale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #blur gray frame for motion detection
            grayBlurFrame = cv2.GaussianBlur(grayFrame, (25, 25), 0)
            #reset firstFrame, whenmotionFrameLimit is hit
            if motionFrameCount == 0:
                motionCompareFrame = grayBlurFrame
            elif motionFrameCount > motionFrameLimit:
                motionFrameCount = 0
        else:#if frame read fails, skip this loop and finish
            frameReadFails += 1 #TODO maybe 1 every video due to logical structure
            continue

        
        motionFrameCount += 1 

        #set json objects for logging json data
        data[frameTotal] = defaultdict(None)
        frameData = data[frameTotal]
        
        #motion detection comparisons
        deltaframe = cv2.absdiff(motionCompareFrame,grayBlurFrame)
        threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold,None)
        countour, heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #loop through detected contours that exceed a certian size
        motionCount = 0
        for i in countour:
            if cv2.contourArea(i) < 1500:
                motionCount += 1
                continue
            #bounding box of the motion
            (mx, my, mw, mh) = cv2.boundingRect(i)
            #draw box on origional frame
            cv2.rectangle(colorFrame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            cv2.putText(colorFrame, 'Motion_' + str(motionCount), (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            #capture bounding box data for eacth motion detected
            motionROIFilePath = fName + '/frame_' + str(frameTotal) + '/motion/motion_' + str(motionCount) + '.jpg'
            frameData['motion_'+str(motionCount)] = {'x': int(mx),'y':int(my), 'w':int(mw), 'h':int(mh),'imgPath': motionROIFilePath}
            
            motionROI = frame[my:my+mh, mx:mx+mw]
            #save detected motion as jpg
            try:
                os.makedirs(fName + '/frame_' + str(frameTotal) + '/motion/' )
                cv2.imwrite(motionROIFilePath, motionROI)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                #if exists, still save file
                try:
                    cv2.imwrite(motionROIFilePath, motionROI)
                except Exception as e:
                    raise
            ###end try catch
            motionCount += 1
        ### end motion loop

        #Look for other stuff, if motion was detected
        if motionCount > 0:
            motionDetected = True #motion detected so we will write the video file

            #look for faces from the front
            faces = faceFrontClassifier.detectMultiScale(grayFrame,1.1,9)
            faceCount = 0
            for (x, y, w, h) in faces:
                #draw box on origional frame
                cv2.rectangle(colorFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(colorFrame, 'Face_' + str(faceCount), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                #capture bounding box data for eacth motion detected
                faceROIFilePath = fName + '/frame_' + str(frameTotal) + '/faces/face_' + str(faceCount) + '.jpg'
                frameData['face_'+str(faceCount)] = {'x': int(x),'y':int(y), 'w':int(w), 'h':int(h),'imgPath': faceROIFilePath}
                #save detected face as jpg
                faceROI = frame[y:y+h, x:x+w]
                try:
                    os.makedirs(fName + '/frame_' + str(frameTotal) + '/faces/' )
                    cv2.imwrite(faceROIFilePath, faceROI)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    #if exists, still save file
                    try:
                        cv2.imwrite(faceROIFilePath, faceROI)
                    except Exception as e:
                        raise
                ###end try catch
                faceCount += 1
            ### end faces loop
            
            #look for face profiles from the front
            faceProfiles = faceProfileClassifier.detectMultiScale(grayFrame,1.1,9)
            faceProfileCount = 0
            for (x, y, w, h) in faceProfiles:
                #draw box on origional frame
                cv2.rectangle(colorFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(colorFrame, 'FaceProfile_' + str(faceProfileCount), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                #capture bounding box data for eacth face profile detected
                faceProfileROIFilePath = fName + '/frame_' + str(frameTotal) + '/faceProfiles/faceProfile_' + str(faceProfileCount) + '.jpg'
                frameData['faceProfile_'+str(faceProfileCount)] = {'x': int(x),'y':int(y), 'w':int(w), 'h':int(h),'imgPath': faceProfileROIFilePath}
                #save detected face as jpg
                faceProfileROI = frame[y:y+h, x:x+w]
                try:
                    os.makedirs(fName + '/frame_' + str(frameTotal) + '/faceProfiles/' )
                    cv2.imwrite(faceProfileROIFilePath, faceProfileROI)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    #if exists, still save file
                    try:
                        cv2.imwrite(faceProfileROIFilePath, faceProfileROI)
                    except Exception as e:
                        raise
                ###end try catch
                faceProfileCount += 1
            ### end faceProfiles loop
            
            #look for cat faces from the front
            catFaces = catFaceFrontClassifier.detectMultiScale(grayFrame,1.1,9)
            catFaceCount = 0
            for (x, y, w, h) in catFaces:
                #draw box on origional frame
                cv2.rectangle(colorFrame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(colorFrame, 'CatFace_' + str(catFaceCount), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                #capture bounding box data for eacth cat face detected
                catFaceROIFilePath = fName + '/frame_' + str(frameTotal) + '/catFaces/catFace_' + str(catFaceCount) + '.jpg'
                frameData['catFace_'+str(catFaceCount)] = {'x': int(x),'y':int(y), 'w':int(w), 'h':int(h),'imgPath': catFaceROIFilePath}
                #save detected cat face as jpg
                catFaceROI = frame[y:y+h, x:x+w]
                try:
                    os.makedirs(fName + '/frame_' + str(frameTotal) + '/catFaces/' )
                    cv2.imwrite(catFaceROIFilePath, catFaceROI)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    #if exists, still save file
                    try:
                        cv2.imwrite(catFaceROIFilePath, catFaceROI)
                    except Exception as e:
                        raise
                ###end try catch
                catFaceCount += 1
            ### end cat faces loop
            #write the frame to jpg file
            cv2.imwrite(fName +'/frame_'+ str(frameTotal) +'/frame'+ str(frameTotal) +'.jpg', colorFrame)
        ### end if motion
        #write frame to video writter
        mp4Vid.write(colorFrame)
    #End frame loop

    #Release video capture object
    vid.release() #release source video file
    
    if motionDetected:
        
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
