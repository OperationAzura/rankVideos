import pika, sys, os
import base64
import cv2
import json
from PIL import Image
from io import BytesIO
import numpy as np

def main():
    connection = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@localhost:5672/'))
    channel = connection.channel()

    channel.queue_declare(queue='frame')

    def callback(ch, method, properties, body):
        print('body: ',type(body))
        jBody = json.loads(body)
        for x in jBody:
            print('x: ',x)
        imdata = base64.b64decode(jBody['frame'])
        im = Image.open(BytesIO(imdata))

        im.save('new.jpg')
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
