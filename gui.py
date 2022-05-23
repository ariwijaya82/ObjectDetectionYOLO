import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import os
import numpy as np
from detection import *

confthres=0.5
nmsthres=0.1

weight_path = os.path.sep.join(['data/', 'yolov4-obj_final.weights'])
config_path = os.path.sep.join(['data/', 'yolov4-obj.cfg'])
label_path = os.path.sep.join(['data/', 'obj.names'])

net = get_net(weight_path, config_path)
classes, colors = get_classes_and_colors(label_path)

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.OffBTN = QPushButton("Stop")
        self.OffBTN.clicked.connect(self.OffFeed)
        self.VBL.addWidget(self.OffBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def OffFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        frame = cv2.imread("data/botol1.jpg")
        detect(frame, net, classes, colors, confthres, nmsthres)
        ConvertToQtFormat = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.ImageUpdate.emit(Pic)

        # self.ThreadActive = True
        # cap = cv2.VideoCapture(2)
        # if not cap.isOpened():
        #     print("Cannot open camera")
        #     exit()

        # while self.ThreadActive:
        #     ret, frame = cap.read()
        #     if not ret:
        #         print("Can't receive frame, Exiting...")
        #         break

        #     height, width = frame.shape[:2]

        #     blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
        #     net.setInput(blob)

        #     output_layers_name = net.getUnconnectedOutLayersNames()
        #     layer_output = net.forward(output_layers_name)

        #     boxes = []
        #     confidances = []
        #     class_ids = []

        #     for output in layer_output:
        #         for detection in output:
        #             score = detection[5:]
        #             class_id = np.argmax(score)
        #             confidance = score[class_id]
        #             if confidance > confthres:
        #                 box = detection[0:4] * np.array([width, height, width, height])
        #                 (centerX, centerY, w, h) = box.astype("int")

        #                 x = int(centerX - w/2)
        #                 y = int(centerY - h/2)

        #                 boxes.append([x,y,w,h])
        #                 confidances.append(float(confidance))
        #                 class_ids.append(class_id)

        #     idxs = cv2.dnn.NMSBoxes(boxes, confidances, confthres, nmsthres)

        #     if len(idxs) > 0:
        #         for i in idxs.flatten():
        #             x, y, w, h = boxes[i]
        #             color = [int(c) for c in colors[class_ids[i]]]
        #             cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        #             text = "{}: {:.4f}".format(classes[class_ids[i]], confidances[i])
        #             cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

        #     ConvertToQtFormat = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        #     Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        #     self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())