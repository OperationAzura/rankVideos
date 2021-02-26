import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


import torchvision.transforms as T
import pika, sys, os
import time
import base64
import cv2
import json
from PIL import Image
from io import BytesIO
import numpy as np

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

#getPrediction gets the bounding boxes and class of the FCNN predictions
def getPrediction(img, threshold):
    start = time.time()
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    predClass = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    predBoxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    predScore = list(pred[0]['scores'].detach().numpy())
    print('predScore: ', predScore)
    print('type: ',type(predScore))
    print('len(): ',len(predScore))
    noMatch = True
    for x in predScore:
        print('xxx: ',x)
        print('threshold: ', threshold )
        if x > threshold:
            noMatch = False
            print('predT: ',predScore.index(x))
    predT = []
    predBoxes = []
    predClass = []
    if not noMatch:
        predT = [predScore.index(x) for x in predScore if x > threshold][-1] # Get list of index with score greater than threshold.
        predBoxes = predBoxes[:predT+1]
        predClass = predClass[:predT+1]
    else:
        print('XXX no matches found?')
    print('!!! '+ str(start - time.time()) + ' seconds to finish!')    
    return predBoxes, predClass


def main():
    connection = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@localhost:5672/'))
    channel = connection.channel()

    channel.queue_declare(queue='frame')

    def callback(ch, method, properties, body):
        start = time.time()
        print('body: ',type(body))
        jBody = json.loads(body)
        for x in jBody:
            print('x: ',x)
        imdata = base64.b64decode(jBody['frame'])
        im = Image.open(BytesIO(imdata))

        threshold = 0.5
        (boxes, classes) = getPrediction(im, threshold)
        response = json.dumps({'boxes': boxes, 'classes': classes, 'fileName':jBody['fileName'], 'frameID': jBody['frameID']})
        #jstr = json.dumps({"frame": base64.b64encode(frameJPG).decode('ascii'), 'fileName': fName, 'frameID': frameTotal})
        #send back results to MQ
        #
        connection = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@localhost:5672/'))
        channel = connection.channel()

        channel.queue_declare(queue='predictedResults')

        channel.basic_publish(exchange='',
                            routing_key='predictedResults',
                            body= response)
        
        connection.close()

        #im.save('new.jpg')
        #cv2.imwrite('./new.jpg', im)
        print(" [x] Received %r" % jBody['fileName'])
        
        #f = open('new.jpg', 'w')
        #cv2.imwrite()

    channel.basic_consume(queue='frame', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
