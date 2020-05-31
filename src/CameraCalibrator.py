from dataclasses import dataclass

import click
import cv2 as cv
import numpy as np

from ChessboardDetector import ChessboardDetector

IMAGE_WIDTH = 1296
IMAGE_HEIGHT = 972


@dataclass
class CameraCalibrator:
    debug: bool = False
    base_folder: str = "calibration/images/"
    n_images: int = 50

    def show_undistorted_images(self, K: np.ndarray, dist: np.ndarray) -> None:
        """
        This method shows three undistorted images to visually verify the quality of
        `K` and `dist`.
        """
        for img_index in range(3):
            img = cv.imread(self.base_folder + f"img_000{img_index:02d}.png")
            img_undistorted = cv.undistort(img, K, dist)
            cv.imshow(f"Undistorted_img_000{img_index:02d}", img_undistorted)
            cv.waitKey()
            cv.destroyAllWindows()

    def compute_intrinsics(self):
        """
        Compute the matrix of intrinsic parameters `K` and the 5 distortion parameters `dist`.
        """
        c = ChessboardDetector(
            base_folder=self.base_folder, n_images=self.n_images, debug=self.debug
        )

        image_points = c.run()
        image_points = np.reshape(
            image_points, (image_points.shape[0], image_points.shape[1], 1, 2)
        )
        object_points = np.array(
            [c.generate_chessboard_corners() for _ in range(image_points.shape[0])]
        )

        print("Calibrating camera...")

        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
            object_points, image_points, (IMAGE_WIDTH, IMAGE_HEIGHT), None, None
        )
        print(f"\n\nRMS: {ret}")
        print(f"\n\nK: {K}")
        print(f"Distortion parameters:\n{dist}")
        print(f"Images used for calibration: {image_points.shape[0]} out of 50")

        Kfile = cv.FileStorage("calibration/intrinsics.xml", cv.FILE_STORAGE_WRITE)
        Kfile.write("RMS", ret)
        Kfile.write("K", K)
        Kfile.write("dist", dist)

        if self.debug:
            self.show_undistorted_images(K, dist)

    def run(self):
        self.compute_intrinsics()


@click.command()
@click.option(
    "--debug", default=False, is_flag=True, help="Use this flag to show debug images"
)
def cli(debug):
    cc = CameraCalibrator(debug=debug)
    cc.run()


if __name__ == "__main__":
    cli()
