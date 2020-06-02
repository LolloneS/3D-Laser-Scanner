import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from src.utils import (
    ExtremePoints,
    Plane,
    Point,
    Rectangle,
    draw_circles,
    fit_plane,
    line_plane_intersection,
    show_image,
    threshold_image,
)


@dataclass
class Scanner3D:
    debug: bool
    K: np.ndarray
    K_inv: np.ndarray
    dist: np.ndarray
    filename: str = "cup1.mp4"
    inner_rectangle: np.ndarray = np.array([[[0, 0]], [[23, 0]], [[23, 13]], [[0, 13]]])
    lower_red_obj: np.ndarray = np.array([35, 25, 40])
    lower_red_planes: np.ndarray = np.array([45, 30, 45])
    upper_red: np.ndarray = np.array([100, 255, 255])
    dbscan: DBSCAN = DBSCAN(eps=7, min_samples=20)

    def get_rectangles_mask(self, thresh: np.ndarray) -> np.ndarray:
        """
        Given a thresholded image of the scene (ideally, the first frame),
        return the masks for the two known rectangles: one on the wall and one on the desk.
        """
        contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
        mask = np.zeros(thresh.shape, np.uint8)
        good_contours = sorted(
            [cnt for cnt in contours if 100000 < cv.contourArea(cnt) < 200000],
            key=cv.contourArea,
        )

        setattr(self, "contour1", good_contours[0])
        setattr(
            self,
            "contour2",
            good_contours[1]
            if cv.pointPolygonTest(
                good_contours[1], tuple(good_contours[0][0][0]), False
            )
            < 0
            else good_contours[2],
        )

        cv.drawContours(mask, [self.contour1], 0, 255, -1)
        cv.drawContours(mask, [self.contour2], 0, 255, -1)

        return mask

    def sort_corners(self, corners: np.ndarray):
        """
        Sort the 4 corners clockwise of a rectangle so that the top-left corner
        is the first one.
        """
        center = np.sum(corners, axis=0) / 4
        sorted_corners = sorted(
            corners,
            key=lambda p: math.atan2(p[0][0] - center[0][0], p[0][1] - center[0][1]),
            reverse=True,
        )
        return np.roll(sorted_corners, 1, axis=0)

    def get_desk_wall_corners(
        self, thresh: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a thresholded image of the scene and a mask representing the two
        known rectangles, return the corners of those rectangles (8 in total)
        with sub-pixel accuracy. The corners returned are already sorted.
        """
        mask = self.get_rectangles_mask(thresh)
        assert thresh.shape[:2] == mask.shape[:2]
        corners = cv.goodFeaturesToTrack(
            thresh,
            maxCorners=8,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask,
            blockSize=5,
        )
        criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 20, 0.001)
        corners = cv.cornerSubPix(
            thresh, corners, winSize=(7, 7), zeroZone=(-1, -1), criteria=criteria
        )
        y_middle = thresh.shape[0] / 2
        desk_corners = np.expand_dims(corners[corners[:, :, 1] > y_middle], axis=1)
        wall_corners = np.expand_dims(corners[corners[:, :, 1] <= y_middle], axis=1)
        sorted_desk_corners = self.sort_corners(desk_corners)
        sorted_wall_corners = self.sort_corners(wall_corners)
        return sorted_desk_corners, sorted_wall_corners

    def get_H_R_t(self, corners: np.ndarray) -> Plane:
        """
        Given 4 sorted corners, compute the homography between the corners
        and the rectangle's ground truth and return the information
        on the mapped plane.
        In other words, this function returns information on a plane
        (in particular, the desk's or wall's).
        The plane's origin is in the top-left corner of the rectangle,
        and the normal is perpendicular to that plane.
        """
        H = cv.findHomography(self.inner_rectangle, corners)[0]
        result = self.K_inv @ H
        result /= cv.norm(result[:, 1])
        r0, r1, t = np.hsplit(result, 3)
        r2 = np.cross(r0.T, r1.T).T
        _, u, vt = cv.SVDecomp(np.hstack([r0, r1, r2]))
        R = u @ vt
        return Plane(origin=t[:, 0], normal=R[:, 2], R=R)

    def get_extreme_points(
        self, wall_corners: np.ndarray, desk_corners: np.ndarray
    ) -> ExtremePoints:
        """
        Given the corners of the rectangles on the wall and on the desk,
        return the coordinates for a tight bounding box of the area
        between the two rectangles.
        """
        ymin_wall = int(np.min(wall_corners[:, :, 1]))
        ymax_wall = int(np.max(wall_corners[:, :, 1]))
        ymin_desk = int(np.min(desk_corners[:, :, 1]))
        ymax_desk = int(np.max(desk_corners[:, :, 1]))
        xmin = int(np.min(wall_corners[:, :, 0]))
        xmax = int(np.max(wall_corners[:, :, 0]))
        return ExtremePoints(
            wall=Rectangle(
                top_left=Point(xmin, ymin_wall), bottom_right=Point(xmax, ymax_wall)
            ),
            desk=Rectangle(
                top_left=Point(xmin, ymin_desk), bottom_right=Point(xmax, ymax_desk)
            ),
        )

    def get_laser_points_in_region(
        self, image: np.ndarray, region: Rectangle, is_obj: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Given an image and a rectangle defining a region, return the laser points
        in that region. In case we are considering the wall or the desk, require
        at least 30 points for better accuracy.
        """
        top_left = region.top_left
        bottom_right = region.bottom_right
        region_image = image[top_left.y : bottom_right.y, top_left.x : bottom_right.x]
        image_inv = cv.cvtColor(~region_image, cv.COLOR_BGR2HSV)
        lower_red = self.lower_red_obj if is_obj else self.lower_red_planes
        red_mask = cv.inRange(image_inv, lower_red, self.upper_red)
        laser_points = cv.findNonZero(red_mask)
        if laser_points is None or (not is_obj and len(laser_points) < 30):
            return None
        return laser_points

    def offset_points(self, points: np.ndarray, offset: Point) -> np.ndarray:
        """Given a region of an image and a point, offset the region by that point."""
        points[:, :, 0] += offset.x
        points[:, :, 1] += offset.y
        return points

    def make_homogeneous(self, points: np.ndarray) -> np.ndarray:
        """
        Given some points, convert them to homogeneous coordinates, i.e. add a trailing [1].
        This function can move points from R^n to P^n, for instance:
        * in R^2: [x y] --> [x y 1]
        * in R^3: [x y z] --> [x y z 1]
        """
        return np.hstack((points[:, 0], np.ones(points.shape[0]).reshape(-1, 1),))

    def remove_obj_outliers(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Use the DBSCAN clustering algorithm in order to remove possible outliers from
        the points detected as laser in the object. We are basically enforcing
        continuity in the laser line on the object, i.e. looking for a dense
        cluster of pixels. Interesting points are the ones whose label is not -1,
        i.e. the ones belonging to a cluster that is not an outlier one.
        """
        dbscan_result = self.dbscan.fit(points[:, 0])
        mask = dbscan_result.labels_ != -1
        return np.expand_dims(points[:, 0][mask], axis=1)

    def get_colors(self, image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Given an image and a list of coordinates of shape (n_points, 1, 2),
        return the RGB colors of those coordinates in the (0...1) range.
        Notice that OpenCV uses BGR instead of RGB by default, thus we need to
        flip the columns.
        """
        x = coordinates.squeeze(1)
        return np.flip(image[x[:, 1], x[:, 0]].astype(np.float64) / 255.0, axis=1)

    def get_laser_points(
        self,
        original_image: np.ndarray,
        image: np.ndarray,
        extreme_points: ExtremePoints,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Given the interesting region of an image, containing the wall and desk planes
        and the object, return the laser points in the three separate regions:
        one for the wall plane, one for the desk plane, one of the object.
        """
        height, width = image.shape[:2]
        ymin_wall = extreme_points.wall.top_left.y
        ymax_wall = extreme_points.wall.bottom_right.y
        ymin_desk = extreme_points.desk.top_left.y
        xmin = extreme_points.desk.top_left.x
        laser_desk = self.get_laser_points_in_region(
            image=image,
            region=Rectangle(
                top_left=Point(0, ymin_desk - ymin_wall),
                bottom_right=Point(width, height),
            ),
        )
        if laser_desk is not None:
            laser_wall = self.get_laser_points_in_region(
                image=image,
                region=Rectangle(
                    top_left=Point(0, 0),
                    bottom_right=Point(width, ymax_wall - ymin_wall),
                ),
            )
            if laser_wall is not None:
                laser_obj = self.get_laser_points_in_region(
                    image=image,
                    region=Rectangle(
                        top_left=Point(0, ymax_wall - ymin_wall),
                        bottom_right=Point(width, ymin_desk - ymin_wall),
                    ),
                    is_obj=True,
                )
                if laser_obj is not None:
                    laser_desk = self.offset_points(
                        points=laser_desk, offset=Point(xmin, ymin_desk)
                    )
                    laser_wall = self.offset_points(
                        points=laser_wall, offset=Point(xmin, ymin_wall)
                    )
                    laser_obj = self.remove_obj_outliers(laser_obj)
                    if laser_obj is not None:
                        laser_obj = self.offset_points(
                            points=laser_obj, offset=Point(xmin, ymax_wall)
                        )
                        obj_colors = self.get_colors(original_image, laser_obj)
                        return laser_wall, laser_desk, laser_obj, obj_colors
        return None, None, None, None

    def save_3d_render(
        self, points: List[np.ndarray], colors: List[np.ndarray]
    ) -> None:
        """
        Given points in the 3D world, save the PLY file representing
        the point cloud. This function saves both the original file and
        a version to which an outlier removal process has been applied.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(points).astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
        if self.debug:
            o3d.visualization.draw_geometries([pcd])
        if not self.debug:
            o3d.io.write_point_cloud(f"results/{self.filename[:-4]}.ply", pcd)

    def read_frame(self, cap) -> Optional[np.ndarray]:
        """
        Read a frame from the cap. Return None if there is no frame left.
        """
        frame_raw = cap.read()[1]
        if frame_raw is None:
            cv.destroyAllWindows()
            return None
        return cv.undistort(frame_raw, self.K, self.dist)

    def create_exiting_rays(
        self, points: np.ndarray, is_obj: bool = False
    ) -> List[np.ndarray]:
        """
        Given a set of 2D points, get their real world 3D coordinates (direction).
        In general, mapping points from the real world [x, y, z, 1]
        to the camera's reference world [x, y, 1] we should multiply
        the real world coordinates by the 3x4 projection matrix P = K[R|T]
        In our case, we want to obtain coordinates in the 3D world starting
        from 2D points in the image, i.e. do the opposite.
        These points are represented in the camera's reference frame:
        this means that R=I and t=[0 0 0]. Only K remains, i.e. the inverse
        operation is done by multiplying each point by K^-1.
        Notice that points and directions, in such a situation, are really tight
        concepts: we can represent a 3D line in space as
        line(λ) = P1 + λ(P1 - P2). Since in this case P2 is the camera center,
        we have that the line is P1 scaled by a factor λ.
        """
        if not is_obj and len(points) > 100:
            points = points[np.random.choice(points.shape[0], 100, replace=False,)]
        return [self.K_inv @ point for point in points]

    def compute_intersections(
        self, plane: Plane, directions: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Given a plane represented by its origin and a normal
        and a list of rays, compute the intersections between
        the plane and the rays.
        """
        return [
            line_plane_intersection(
                plane_origin=plane.origin,
                plane_normal=plane.normal,
                line_direction=direction,
            )
            for direction in directions
        ]

    def run(self):
        cap = cv.VideoCapture(f"videos/{self.filename}")
        if not cap.isOpened():
            return
        first_frame = self.read_frame(cap)
        if first_frame is None:
            return
        first_frame_thresh = threshold_image(first_frame)
        desk_corners, wall_corners = self.get_desk_wall_corners(first_frame_thresh)
        extreme_points = self.get_extreme_points(wall_corners, desk_corners)

        desk_plane = self.get_H_R_t(desk_corners)
        wall_plane = self.get_H_R_t(wall_corners)

        if self.debug:
            first_frame_copy = first_frame.copy()
            cv.drawContours(first_frame_copy, [self.contour1], -1, (255, 255, 255), 2)
            cv.drawContours(first_frame_copy, [self.contour2], -1, (255, 255, 255), 2)
            draw_circles(first_frame_copy, desk_corners, text=True)
            draw_circles(first_frame_copy, wall_corners, text=True)
            show_image(first_frame_copy)

        all_obj_points = []
        all_obj_colors = []
        while True:
            frame = self.read_frame(cap)
            if frame is None:
                break
            frame_copy = frame.copy()
            frame_interesting = frame[
                extreme_points.wall.top_left.y : extreme_points.desk.bottom_right.y,
                extreme_points.wall.top_left.x : extreme_points.wall.bottom_right.x,
            ]
            (laser_wall, laser_desk, laser_obj, obj_colors,) = self.get_laser_points(
                first_frame, frame_interesting, extreme_points
            )
            if laser_wall is not None:
                if self.debug:
                    draw_circles(frame_copy, laser_wall)
                    draw_circles(frame_copy, laser_desk)
                    draw_circles(frame_copy, laser_obj)

                laser_wall = self.make_homogeneous(laser_wall)
                laser_obj = self.make_homogeneous(laser_obj)
                laser_desk = self.make_homogeneous(laser_desk)

                wall_directions = self.create_exiting_rays(laser_wall, is_obj=False)
                desk_directions = self.create_exiting_rays(laser_desk, is_obj=False)
                obj_directions = self.create_exiting_rays(laser_obj, is_obj=True)

                intersections_wall = self.compute_intersections(
                    wall_plane, wall_directions
                )
                intersections_desk = self.compute_intersections(
                    desk_plane, desk_directions
                )
                intersections_rects = np.array(intersections_wall + intersections_desk)
                laser_plane = fit_plane(intersections_rects)
                intersections_objs = self.compute_intersections(
                    laser_plane, obj_directions
                )
                all_obj_points.extend(intersections_objs)
                all_obj_colors.extend(obj_colors)

            if self.debug:
                if show_image(frame_copy, continuous=True):
                    break
            else:
                if show_image(frame, continuous=True):
                    break

        self.save_3d_render(all_obj_points, all_obj_colors)
        cap.release()
        cv.destroyAllWindows()
