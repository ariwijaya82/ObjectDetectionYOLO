import sys
import os
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
import cv2

from detection import *

class ImageWorker(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        weight_path = os.path.sep.join(['data/', 'training.weights'])
        config_path = os.path.sep.join(['data/', 'obj.cfg'])
        label_path = os.path.sep.join(['data/', 'obj.names'])
        video_path = os.path.sep.join(['data/', 'test.mp4'])
        image_path = os.path.sep.join(['data/', 'test.jpg'])

        confthres=0.5
        nmsthres=0.1

        net = get_net(weight_path, config_path)
        classes, colors = get_classes_and_colors(label_path)

        flag = False

        if flag:
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
                    detect(frame, net, classes, colors, confthres, nmsthres)
                else:
                    sleep(0.05)

                ConvertToQtFormat = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

        else:
            image = cv2.imread(image_path)
            detect(image, net, classes, colors, confthres, nmsthres)
            ConvertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
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
        
        # temp
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(False)

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