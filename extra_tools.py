from dataclasses import dataclass
from math import degrees, atan2
from typing import Tuple
import cv2
import numpy as np


class ExtraTools:

    @staticmethod
    def merge_images(img: np.array, img_background: np.array):
        """
        :param img: image you want to put on the 2nd image
        :param img_background: your main image (2nd image)
        :return:
        """
        alpha_channel = img[:, :, 3] / 255  # convert from 0-255 to 0.0-1.0
        overlay_colors = img[:, :, :3]
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
        h, w = img.shape[:2]
        background_subsection = img_background[0:h, 0:w]

        # combine the background with the overlay image weighted by alpha
        final_img = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

        return final_img

    @staticmethod
    def rotate(image: np.array, angle: float, center=None, scale=1.0) -> np.array:
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    @staticmethod
    def angle3pt(a, b, c):
        """Counterclockwise angle in degrees by turning from a to c around b
            Returns a float between 0.0 and 360.0"""
        ang = degrees(atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 180 if ang < 0 else ang

    @staticmethod
    def angle2pt(a, b):
        ang = degrees(atan2(b[1] - a[1], b[0] - a[0]))
        return ang


@dataclass
class Button(ExtraTools):
    image_path: str
    size: Tuple[int, int] = (128, 128)

    def __post_init__(self):
        self.button_img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        self.button_img = cv2.resize(self.button_img, self.size)

    def draw_button(self, img: np.array, button_roi: Tuple[int, int, int, int], flip: bool = True) -> None:
        if flip:
            self.button_img = cv2.flip(self.button_img, -1)

        y1, y2, x1, x2 = button_roi
        button_roi = img[y1:y2, x1:x2]

        button = self.merge_images(self.button_img, button_roi)
        img[y1:y2, x1:x2] = button
