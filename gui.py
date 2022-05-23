import sys
import os
import cv2
from time import sleep

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal
)
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from detection import *

class ImageWorker(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        weight_path = os.path.sep.join(['data/', 'final.weights'])
        config_path = os.path.sep.join(['data/', 'obj.cfg'])
        label_path = os.path.sep.join(['data/', 'obj.names'])
        video_path = os.path.sep.join(['data/', 'test.mp4'])

        confthres=0.5
        nmsthres=0.1

        net = get_net(weight_path, config_path)
        classes, colors = get_classes_and_colors(label_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("cannot open media")
            sys.exit()

        self.startClassification = False
        while True:
            _, frame = cap.read()
            if not _:
                print("Can't receive frame, Exiting...")
                break

            if self.startClassification:
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

            else:
                sleep(0.05)

            ConvertToQtFormat = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.clickCount = 0
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("X-Ray Detection")
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.image = QLabel()
        self.imageWorker = ImageWorker()
        self.imageWorker.start()
        self.imageWorker.ImageUpdate.connect(self.update)

        self.btnStart = QPushButton("Start")
        self.btnStart.clicked.connect(self.start)

        self.btnStop = QPushButton("Stop")
        self.btnStop.clicked.connect(self.stop)

        layout = QVBoxLayout()
        layout.addWidget(self.image)
        layout.addWidget(self.btnStart)
        layout.addWidget(self.btnStop)
        self.centralWidget.setLayout(layout)

    def update(self, Image):
        self.image.setPixmap(QPixmap.fromImage(Image))

    def start(self):
        self.imageWorker.startClassification = True
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)

    def stop(self):
        self.imageWorker.startClassification = False
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())