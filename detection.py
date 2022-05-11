import cv2
import numpy as np

def get_classes_and_colors(path):
    classes = []
    with open(path, 'r') as f:
        classes = f.read().splitlines()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    return classes, colors

def get_net(weight_path, config_path):
    net = cv2.dnn.readNet(weight_path, config_path)
    return net

def detect(frame, net, classes, colors, confthres, nmsthres):
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()
    layer_output = net.forward(output_layers_name)

    boxes = []
    confidances = []
    class_ids = []

    for output in layer_output:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidance = score[class_id]
            if confidance > confthres:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                x = int(centerX - w/2)
                y = int(centerY - h/2)

                boxes.append([x,y,w,h])
                confidances.append(float(confidance))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidances, confthres, nmsthres)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidances[i])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)