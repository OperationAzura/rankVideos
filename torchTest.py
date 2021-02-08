import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

import torchvision.transforms as T
import cv2
from PIL import Image

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

def get_prediction(img_path, threshold):
    img = Image.open(img_path) # Load the image
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection_api(count, img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    
    boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    
    cv2.imwrite('torchTest' + str(count)+'.jpg', img)
    #
    #
    #
    #plt.figure(figsize=(20,30)) # display the output image
    #plt.imshow(img)
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()


if __name__ == "__main__":
    print('starting torch test')
    path = '/home/derek/projects/rankVideos'
    filePaths = ['/CarPort_2021_01_29_162848.mp4/frame_1490/frame1490.jpg',
        '/CarPort_2021_01_30_141116.mp4/frame_1020/frame1020.jpg',
        '/CarPort_2021_01_30_093834.mp4/frame_69/frame69.jpg',
        '/CarPort_2021_01_31_134206.mp4/frame_1582/frame1582.jpg',
        '/CarPort_2021_01_31_134206.mp4/frame_1591/frame1591.jpg',
        '/CarPort_2021_01_31_134206.mp4/frame_1579/frame1579.jpg',
        '/CarPort_2021_01_31_134206.mp4/frame_1581/frame1581.jpg',
        '/CarPort_2021_01_31_134206.mp4/frame_1583/frame1583.jpg',
        '/CarPort_2021_01_29_085159.mp4/frame_13/frame13.jpg',
        '/CarPort_2021_01_29_085159.mp4/frame_14/frame14.jpg',
        '/CarPort_2021_01_29_085159.mp4/frame_430/frame430.jpg']
    count = 0
    for f in filePaths:
        print('attempting file: ', f)
        object_detection_api(count, path + f , rect_th=6, text_th=5, text_size=5)
        count += 1
    print('finished ranking')
