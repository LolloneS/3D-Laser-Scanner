import math
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from utils import draw_circles, show_image, threshold_image


@dataclass
class ChessboardDetector:
    base_folder: str = "calibration/images/"
    n_images: int = 50
    destination: np.array = np.array([[[0, 900]], [[0, 0]], [[700, 0]], [[700, 900]]])
    debug: bool = False
    width: int = 700
    height: int = 900

    def generate_chessboard_corners(self):
        """
        Generate the 3D objectPoints vector needed for `calibrateCamera`.
        This is generated programmatically as it reproduces the SVG chessboard.
        """
        corners = (
            [[1, j, 0] for j in range(1, 12, 2)]
            + [[i, j, 0] for i in range(3, 14, 2) for j in range(1, 14, 2)]
            + [[15, j, 0] for j in range(1, 12, 2)]
            + [[17, j, 0] for j in range(3, 10, 2)]
        )
        return np.expand_dims(np.array(corners), axis=1).astype("float32")

    def get_chessboard_mask(self, thresh):
        """
        Given a B/W thresholded image `thresh`,
        return a mask which contains the chessboard and removes basic noise (such as X-Y symbols).
        The "internal" rectangle containing the chessboard is always the third one (after having
        sorted by area in descending order).
        """
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(thresh.shape, np.uint8)
        areas = sorted(
            [i for i in range(len(contours))],
            key=lambda i: cv.contourArea(contours[i]),
            reverse=True,
        )
        chessboard = contours[areas[2]]
        cv.drawContours(mask, [chessboard], 0, 255, -1)
        location_xy = None
        for cnt in contours:
            if (
                5 < cv.contourArea(cnt) < 1000
                and cv.pointPolygonTest(chessboard, tuple(cnt[0][0]), True) > 1
            ):
                location_xy = cnt[0].astype("float32")
                return mask, chessboard, location_xy

    def find_closest_point(self, points, target):
        """
        Given `n` points and a target point,
        return the point which is closest to the target
        """
        index = 0
        minimum = points[0]
        min_dist = cv.norm(target, points[0])
        for i, point in enumerate(points[1:]):
            new_dist = cv.norm(target, point)
            if new_dist < min_dist:
                min_dist = new_dist
                minimum = point
                index = i + 1
        return minimum, index

    def load_and_threshold_image(self, filename: str):
        """
        Given a path to an image, load it, threshold it and return the two results.
        """
        img = cv.imread(filename)
        img_thresholded = threshold_image(img)
        return img, img_thresholded

    def find_four_corners(self, mask: np.ndarray) -> np.ndarray:
        """
        Given a mask quite similar to a rectangle, return the four corners.
        """
        chessboard_corners = cv.goodFeaturesToTrack(
            mask,
            maxCorners=4,
            qualityLevel=0.05,
            minDistance=100,
            mask=None,
            blockSize=5,
            gradientSize=3,
        )
        criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 20, 0.001)
        chessboard_corners = cv.cornerSubPix(
            mask,
            chessboard_corners,
            winSize=(3, 3),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        return chessboard_corners

    def get_sorted_corners(self, chessboard_corners, location_xy):
        """
        Given four corners and the location of the "XY" sign, sort the corners in clockwise order,
        with the first corner being the one closer to the XY sign.
        """
        corner_0_0, index_0_0 = self.find_closest_point(chessboard_corners, location_xy)
        center = np.sum(chessboard_corners, axis=0) / 4
        sorted_corners = sorted(
            chessboard_corners,
            key=lambda p: math.atan2(p[0][0] - center[0][0], p[0][1] - center[0][1]),
            reverse=True,
        )
        while any(sorted_corners[0].ravel() != corner_0_0.ravel()):
            sorted_corners = np.roll(sorted_corners, 1)
        return sorted_corners

    def create_inner_mask(self, shape):
        """
        Given the shape of the straightened chessboard, create the mask.
        This removes 30px per border, as the actual squares of the chessboard start ~50 pixels
        from the border by construction.
        """
        mask = np.full(shape, 255, np.uint8)
        s1 = shape[1]
        s0 = shape[0]
        cv.rectangle(mask, (0, 0), (s1, 30), 0, -1)
        cv.rectangle(mask, (0, 0), (30, s0), 0, -1)
        cv.rectangle(
            mask, (0, s0 - 30), (s1, s0), 0, -1,
        )
        cv.rectangle(
            mask, (s1 - 30, 0), (s1, s0), 0, -1,
        )
        return mask

    def sort_inner_corners(self, inner_corners):
        """
        Given a `warped` rectified image of the chessboard's inner corners,
        sort the corners starting from the XY symbol on the bottom left of the chessboard.
        """
        sorted_corners = []
        for i in np.arange(start=self.height, stop=0, step=-100):
            sorted_corners += sorted(
                inner_corners[
                    np.logical_and(
                        inner_corners[:, :, 1] < i, inner_corners[:, :, 1] > i - 100
                    )
                ],
                key=lambda p: p[0],
                reverse=False,
            )
        return np.array(sorted_corners)

    def find_inner_corners_warped(self, warped_thresholded: np.ndarray):
        """
        Given a thresholded `warped` image obtained via homography, get the inner corners of the chessboard.
        """
        contours = cv.findContours(
            warped_thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )[0]
        mask = self.create_inner_mask(warped_thresholded.shape)

        for cnt in contours:
            if cv.contourArea(cnt) < 1000:
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

        # smooth the edges a tad
        kernel = np.ones((5, 5), np.uint8)
        opening = cv.morphologyEx(warped_thresholded, cv.MORPH_OPEN, kernel)

        inner_corners = cv.goodFeaturesToTrack(
            opening,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=85,
            mask=mask,
            blockSize=3,
            gradientSize=9,
        )
        criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 20, 0.001)
        inner_corners = cv.cornerSubPix(
            opening, inner_corners, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
        )

        if self.debug and self.img_index % 10 == 0:
            opening_copy = opening.copy()
            for corner in inner_corners:
                cv.circle(
                    img=opening_copy,
                    center=tuple(corner[0]),
                    radius=8,
                    color=(130, 255, 155),
                    thickness=-1,
                )
            show_image(opening_copy)

        sorted_corners = self.sort_inner_corners(inner_corners)
        return np.expand_dims(sorted_corners, axis=1).astype("float32")

    def run(self):
        result = []
        setattr(self, "img_index", 0)
        for self.img_index in range(self.n_images):
            img, img_thresholded = self.load_and_threshold_image(
                self.base_folder + f"img_000{self.img_index:02d}.png"
            )
            mask, chessboard, location_xy = self.get_chessboard_mask(img_thresholded)

            if self.debug and self.img_index % 10 == 0:
                img_copy = img.copy()
                cv.drawContours(img_copy, [chessboard], -1, (236, 255, 122), 5)

            chessboard_corners = self.find_four_corners(mask)
            sorted_corners = self.get_sorted_corners(chessboard_corners, location_xy)

            if self.debug and self.img_index % 10 == 0:
                draw_circles(img_copy, sorted_corners)
                show_image(img_copy)

            H = cv.findHomography(np.array(sorted_corners), self.destination)[0]
            H_inv = np.linalg.inv(H)
            warped = cv.warpPerspective(img, H, (700, 900))
            warped_thresh = threshold_image(warped)
            inner_corners = self.find_inner_corners_warped(warped_thresh)
            inner_corners_in_original_image = []
            for i, corner in enumerate(inner_corners):
                corner_inv = H_inv @ np.vstack([corner[0][0], corner[0][1], [1]])
                corner_inv = corner_inv[:2] / corner_inv[2]
                inner_corners_in_original_image.append([corner_inv])

            if self.debug and self.img_index % 10 == 0:
                draw_circles(img_copy, inner_corners_in_original_image, text=True)
                show_image(img_copy)
            result = result + [inner_corners_in_original_image]
        return np.array(result).astype("float32")
