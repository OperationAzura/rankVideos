import json
from drawConfigClass import DrawConfig

class ConfigClass(object):
    def __init__(self, filePath="config.json", knownFaceNamesPath="ref_name.pkl", knownFaceEncodingsPath="ref_embed.pkl", cvDataPath="torchData.json", logPath="log.log", imagePaths=["./"], baseOutputPath="torch/images/", scalePercent=50, rectThickness=3, textThickness=3, textSize=3, threshold=0.5):
        self.filePath = filePath
        self.knownFaceNamesPath = knownFaceNamesPath
        self.knownFaceEncodingsPath = knownFaceEncodingsPath
        self.cvDataPath = cvDataPath
        self.logPath = logPath
        self.imagePaths = imagePaths
        self.baseOutputPath = baseOutputPath
        self.scalePercent = scalePercent
        self.drawConfig = DrawConfig(rectThickness, textThickness, textSize, threshold)
        self.load()



        pass
    
    #load will load the config from a file
    def load(self ):
        #self.filePath = filePath #accept new file path if given
        j = None
        try:
            j = None
            with open(self.filePath, 'r') as f:
                j = json.loads(f.read())
                f.close()
        except:
            print('didnt load config')

        if j["knownFaceNamesPath"] != None:
            self.knownFaceNamesPath = j["knownFaceNamesPath"]
        if j["knownFaceEncodingsPath"] != None:
            self.knownFaceEncodingsPath = j["knownFaceEncodingsPath"]
        if j["cvDataPath"] != None:
            self.cvDataPath = j["cvDataPath"]
        if j["logPath"] != None:
            self.logPath = j["logPath"]
        if j["imagePaths"] != None:
            self.imagePaths = j["imagePaths"]
        if j["baseOutputPath"] != None:
            self.baseOutputPath = j["baseOutputPath"]
        if j["scalePercent"] != None:
            self.scalePercent = j["scalePercent"]
        if j["drawConfig"] != None:
            #check sub-config items
            dConf = j["drawConfig"]
            if dConf["rectThickness"] != None:
                self.drawConfig.rectThickness = dConf["rectThickness"]
            if dConf["textThickness"] != None:
                self.drawConfig.textThickness = dConf["textThickness"]
            if dConf["textSize"] != None:
                self.drawConfig.textSize = dConf["textSize"]
            if dConf["threshold"] != None:
                self.drawConfig.threshold = dConf["threshold"]
        else: #initialize drawConfig
            self.drawConfig = DrawConfig()
        
