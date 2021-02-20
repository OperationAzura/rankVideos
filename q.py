import pika

connection = pika.BlockingConnection(pika.URLParameters('amqp://derek:bazinga1@localhost:5672/'))
channel = connection.channel()

channel.queue_declare(queue='hello')

channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Hello World!')
print(" [x] Sent 'Hello World!'")
connection.close()

