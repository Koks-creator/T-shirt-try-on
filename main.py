from dataclasses import dataclass
import glob
from time import time
from math import hypot, degrees, atan2
from typing import NamedTuple
import cv2
import mediapipe as mp
import numpy as np

from ShirtTryOn.extra_tools import ExtraTools, Button


@dataclass
class MpPoseUtils:
    static_image_mode: bool = False,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    enable_segmentation: bool = False,
    smooth_segmentation: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5

    def __post_init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def preprocess_img(self, img: np.array):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        results = self.pose.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img, results

    def get_landmarks(self, img: np.array, pose_results: NamedTuple, draw=True):
        landmarks = []

        if pose_results.pose_landmarks:
            h, w, _ = img.shape
            if draw:
                self.mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            for lm in pose_results.pose_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append([x, y, lm.z])

        return landmarks


RIGHT_SHOULDER = 11
RIGHT_HIP = 23
RIGHT_ELBOW = 13
LEFT_SHOULDER = 12
LEFT_HIP = 24
LEFT_ELBOW = 14
WIDTH_OFFSET = 100
HEIGHT_OFFSET = 120
LEFT_BUTTON_CENTER = (100, 300)
WINDOW_SHAPE = (1280, 720)
ELLIPSE_COLOR = (255, 255, 255)
BUTTON_PATH = "button.png"
T_SHIRTS_PATH_PATTERN = "tshirts/*.png"

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Videos/Pexels Videos 2785536.mp4")
mp_pose_util = MpPoseUtils()
tools = ExtraTools()

radius = 80
axes = (radius, radius)
angle_left_btn = 0
start_angle_left_btn = 0
end_angle_left_btn = 0

angle_right_btn = 0
start_angle_right_btn = 0
end_angle_right_btn = 0

left_btn_chosen = False
right_btn_chosen = False

t_shirt_images = glob.glob(T_SHIRTS_PATH_PATTERN)
shirt_index = 0
p_time = 0

while cap.isOpened():
    success, frame = cap.read()
    # frame = cv2.flip(frame, 1)

    if not success:
        break

    frame = cv2.resize(frame, WINDOW_SHAPE)

    frame, results = mp_pose_util.preprocess_img(frame)
    lms = mp_pose_util.get_landmarks(frame, results)

    if lms:
        left_shoulder = lms[LEFT_SHOULDER]
        left_hip = lms[LEFT_HIP]
        left_elbow = lms[LEFT_ELBOW]

        right_shoulder = lms[RIGHT_SHOULDER]
        right_hip = lms[RIGHT_HIP]
        right_elbow = lms[RIGHT_ELBOW]

        z1 = round(left_shoulder[2], 3)
        z2 = round(right_shoulder[2], 3)
        z_diff = round(z1 - z2, 3)
        z_diff = z_diff * -1 if z_diff < 0 else z_diff

        if z_diff < 0.3:
            if left_btn_chosen and not right_btn_chosen:
                cv2.ellipse(frame, (112, 332), axes, angle_left_btn, start_angle_left_btn, end_angle_left_btn,
                            ELLIPSE_COLOR, cv2.FILLED)

                start_angle_left_btn += 20

                if start_angle_left_btn == 360:
                    start_angle_left_btn = 0
                    shirt_index -= 1

                    if shirt_index < 0:
                        shirt_index = len(t_shirt_images) - 1
            else:
                start_angle_left_btn = 0

            if right_btn_chosen and not left_btn_chosen:
                cv2.ellipse(frame, (1162, 332), axes, angle_right_btn, start_angle_right_btn, end_angle_right_btn,
                            ELLIPSE_COLOR, cv2.FILLED)

                start_angle_right_btn += 20

                if start_angle_right_btn == 360:
                    start_angle_right_btn = 0
                    shirt_index += 1

                    if shirt_index >= len(t_shirt_images):
                        shirt_index = 0
            else:
                start_angle_right_btn = 0

            # Setup user interface
            button_left = Button(BUTTON_PATH)
            button_left.draw_button(frame, (270, 270 + 128, 50, 50 + 128))

            button_right = Button(BUTTON_PATH)
            button_right.draw_button(frame, (270, 270 + 128, 1100, 1100 + 128), flip=False)

            left_arm_ang = abs(tools.angle3pt(left_hip, left_elbow, left_shoulder))
            right_arm_ang = abs(tools.angle3pt(right_hip, right_elbow, right_shoulder))

            if left_arm_ang > 95:
                left_btn_chosen = True
            else:
                left_btn_chosen = False

            if right_arm_ang > 95:
                right_btn_chosen = True
            else:
                right_btn_chosen = False

            t_shirt_img = cv2.imread(t_shirt_images[shirt_index], cv2.IMREAD_UNCHANGED)
            t_shirt_h, t_shirt_w, _ = t_shirt_img.shape
            h2w_ratio = t_shirt_h / t_shirt_w

            x1 = left_shoulder[0]
            y1 = left_shoulder[1]
            x2 = right_shoulder[0]
            y2 = right_hip[1]

            scale = round(abs(left_shoulder[0] - right_shoulder[0]) / t_shirt_w, 2)
            current_offset = int(WIDTH_OFFSET * scale), int(HEIGHT_OFFSET * scale)

            x1 = x1 - current_offset[0]
            x2 = x2 + current_offset[0]
            y1 = y1 - current_offset[1]
            y2 = y2 + current_offset[1]

            roi_center_top = left_shoulder[0] + (abs(right_shoulder[0] - left_shoulder[0]) // 2), left_shoulder[1]
            roi_center_bottom = left_hip[0] + (abs(right_hip[0] - left_hip[0]) // 2), left_hip[1]

            ang_org = degrees(atan2(roi_center_top[0] - roi_center_bottom[0], roi_center_top[1] - roi_center_bottom[1]))
            ang = 180 - (ang_org * -1)
            t_shirt_img = tools.rotate(t_shirt_img, ang)

            t_shirt_width = int(hypot(x1 - x2, left_hip[1] - right_hip[1]))
            t_shirt_height = int(t_shirt_width * h2w_ratio)

            roi_x1 = x1
            roi_x2 = x1 + t_shirt_width
            roi_y1 = y1
            roi_y2 = y1 + t_shirt_height

            # Make it look better when angled
            if 220 < ang < 355:
                x1 = left_shoulder[0]-20
                y1 = left_shoulder[1]+20
                x2 = right_shoulder[0]
                y2 = right_hip[1]

                x1 = x1 - current_offset[0]
                x2 = x2 + current_offset[0]
                y1 = y1 - current_offset[1]
                y2 = y2 + current_offset[1]

                roi_x1 = x1
                roi_x2 = x1 + t_shirt_width
                roi_y1 = y1
                roi_y2 = y1 + t_shirt_height

            if 10 < ang < 55:
                x1 = left_shoulder[0]
                y1 = left_hip[1]
                x2 = int(right_shoulder[0]+20)
                y2 = int(right_shoulder[1]+20)

                x1 = x1 - current_offset[0]
                x2 = x2 + current_offset[0]
                y1 = y1 + current_offset[1]
                y2 = y2 - current_offset[1]

                roi_x1 = x2 - t_shirt_width
                roi_x2 = x2
                roi_y1 = y2
                roi_y2 = y2 + t_shirt_height

            roi = frame[roi_y1:roi_y2,
                        roi_x1:roi_x2]

            t_shirt_img = cv2.resize(t_shirt_img, (t_shirt_width, t_shirt_height))
            try:
                final_img = tools.merge_images(t_shirt_img, roi)
                frame[roi_y1:roi_y2,
                      roi_x1:roi_x2] = final_img
            except ValueError:
                pass
        else:
            cv2.putText(frame, "You need to face camera", (400, 400), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 200), 2)

    c_time = time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time

    cv2.putText(frame, f"FPS: {fps}", (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 200, 200), 2)

    cv2.imshow("Res", frame)
    # cv2.imshow("Resc", roi)
    # cv2.imshow("Reswwc", t_shirt_img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
