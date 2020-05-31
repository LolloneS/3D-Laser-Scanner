import click
import numpy as np

from src.Scanner3D import Scanner3D
from src.utils import load_intrinsics


@click.command()
@click.option("--debug", default=False, is_flag=True, help="Show debug images")
@click.option(
    "--filename", default="cup1.mp4", help="Name of the video to be processed"
)
def cli(debug, filename):
    K, dist = load_intrinsics(debug=debug)
    s = Scanner3D(
        filename=filename, K=K, dist=dist, K_inv=np.linalg.inv(K), debug=debug
    )
    s.run()


if __name__ == "__main__":
    cli()
