from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np
import time
import PoseModule as pm

class AITrainerApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trainer")
        self.setGeometry(100, 100, 1280, 800)

        # Set up the main layout
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        # Video Display Area
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(1280, 720)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label)

        # Curl Counter Display
        self.counter_label = QtWidgets.QLabel("Curls: 0")
        self.counter_label.setFont(QtGui.QFont("Arial", 24, QtGui.QFont.Bold))
        self.counter_label.setStyleSheet("color: green;")
        self.counter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.counter_label)

        # Control Buttons Layout
        self.button_layout = QtWidgets.QHBoxLayout()
        self.upload_button = QtWidgets.QPushButton("Upload Video")
        self.start_button = QtWidgets.QPushButton("Start Webcam")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.button_layout.addWidget(self.upload_button)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.pause_button)
        self.button_layout.addWidget(self.reset_button)
        self.layout.addLayout(self.button_layout)

        # Styling for buttons
        button_style = """
        QPushButton {
            font-size: 18px;
            padding: 10px;
            background-color: #5A9;
            border-radius: 10px;
            color: white;
        }
        QPushButton:hover {
            background-color: #7AB;
        }
        """
        self.upload_button.setStyleSheet(button_style)
        self.start_button.setStyleSheet(button_style)
        self.pause_button.setStyleSheet(button_style)
        self.reset_button.setStyleSheet(button_style)

        # Initialize variables
        self.running = False
        self.cap = None
        self.detector = pm.poseDetector()
        self.count = 0
        self.dir = 0
        self.pTime = 0
        self.video_path = None

        # Connect buttons
        self.upload_button.clicked.connect(self.upload_video)
        self.start_button.clicked.connect(self.start_webcam)
        self.pause_button.clicked.connect(self.pause)
        self.reset_button.clicked.connect(self.reset)

        # Timer for updating the video
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def upload_video(self):
        # Open file dialog to select video
        self.video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4 *.avi)")
        if self.video_path:
            self.start_video()

    def start_video(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.start(30)

    def start_webcam(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)

    def pause(self):
        if self.running:
            self.running = False
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def reset(self):
        self.count = 0
        self.dir = 0
        self.counter_label.setText("Curls: 0")
        self.pause()

    def update_frame(self):
        success, img = self.cap.read()
        if not success:
            self.pause()
            return

        img = cv2.resize(img, (1280, 720))
        img = self.detector.findPose(img, False)
        lmList = self.detector.findPosition(img, False)

        if len(lmList) != 0:
            # Right Arm
            angle = self.detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))

            # Check for the dumbbell curls
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if self.dir == 0:
                    self.count += 0.5
                    self.dir = 1
            if per == 0:
                color = (0, 255, 0)
                if self.dir == 1:
                    self.count += 0.5
                    self.dir = 0

            # Update counter label
            self.counter_label.setText(f"Curls: {int(self.count)}")

            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # Draw Curl Count in a larger font size
            cv2.putText(
                img,
                f'Count: {int(self.count)}',
                (50, 670),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,  # Increased font scale for larger text
                (255, 0, 0),
                10  # Increased thickness for better visibility
            )

        # Convert image to RGB format for Qt display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(convert_to_qt_format)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.pause()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = AITrainerApp()
    window.show()
    app.exec_()
