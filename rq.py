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
import logging

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

#logging setup
logging.basicConfig(filename='rq.log', level=logging.INFO)

#getPrediction gets the bounding boxes and class of the FCNN predictions
def getPrediction(img, threshold):
    logging.info('!!!getPrediction starting')
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
        logging.info('XXX no matches found?')
    logging.info('!!!getPrediction finished in: ' + str(start - time.time()))
    return predBoxes, predClass


def main():
    mqURL = 'amqp://derek:bazinga1@192.168.1.12:5672/'
    frameQueue = 'frame'
    predictedQueue = 'predictedResults'
    connection = pika.BlockingConnection(pika.URLParameters(mqURL))
    channel = connection.channel()

    channel.queue_declare(queue=frameQueue)

    def callback(ch, method, properties, body):
        logging.info('callback fired')
        start = time.time()
        jBody = json.loads(body)
        imdata = base64.b64decode(jBody['frame'])
        im = Image.open(BytesIO(imdata))

        threshold = 0.5
        (boxes, classes) = getPrediction(im, threshold)
        response = json.dumps({'boxes': boxes, 'classes': classes, 'fileName':jBody['fileName'], 'frameID': jBody['frameID']})
        resCon = pika.BlockingConnection(pika.URLParameters(mqURL))
        resChan = resCon.channel()
        resChan.queue_declare(queue=predictedQueue)
        resChan.basic_publish(exchange='',
                            routing_key=predictedQueue,
                            body= response)
        
        resCon.close()

        #im.save('new.jpg')
        #cv2.imwrite('./new.jpg', im)
        print(" [x] Received %r" % jBody['fileName'])
        
        #f = open('new.jpg', 'w')
        #cv2.imwrite()
        print('callback ended in: ' + str(time.time() - start) )
        logging.info('callback ended in: ' + str(time.time() - start) )

    channel.basic_consume(queue=frameQueue, on_message_callback=callback, auto_ack=True)

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
