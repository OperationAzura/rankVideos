#DrawConfig is a piece of the config class
class DrawConfig(object):
    def __init__(self, rectThickness=3, textThickness=3, textSize=3, threshold=0.5):
        self.rectThickness = rectThickness
        self.textThickness = textThickness
        self.textSize = textSize
        self.threshold = threshold
