import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from typing import List

# Take in detection pickle file image folder path as command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    '--detections_pickle',
                    type=str,
                    required=True,
                    help='Path to detections pickle file')
parser.add_argument('-i',
                    '--image_folder',
                    type=str,
                    required=True,
                    help='Path to image folder')
args = parser.parse_args()

detections_pickle = Path(args.detections_pickle)
assert detections_pickle.is_file(), 'Detections pickle file not found'

image_folder = Path(args.image_folder)
assert image_folder.is_dir(), 'Image folder not found'


def get_image_paths(image_folder: Path) -> list:
    """
    Get all images in image folder ending in .jpg or .png
    """
    suffix_set = {'.jpg', '.png'}
    return [
        image for image in image_folder.iterdir() if image.suffix in suffix_set
    ]


def get_detections(detections_pickle: Path) -> dict:
    """
    Get data from detections pickle file
    """
    with open(detections_pickle, 'rb') as f:
        data = pickle.load(f)
    return data


image_paths = get_image_paths(image_folder)
detections_dict = get_detections(detections_pickle)


def draw_rectangle_confidence(rect_points: List[int], confidence: float):
    """
    Draw rectangle with confidence
    """
    x0, y0 = rect_points[0:2]
    width = rect_points[2] - x0
    height = rect_points[3] - y0
    plt.gca().add_patch(
        plt.Rectangle((x0, y0),
                      width,
                      height,
                      fill=False,
                      edgecolor='red',
                      linewidth=2))
    plt.gca().text(x0,
                   y0 + 9,
                   '{:.2f}'.format(confidence),
                   bbox={
                       'facecolor': 'white',
                       'alpha': 0.5,
                       'pad': 1
                   })


for p in image_paths:
    image_name = p.stem
    assert image_name in detections_dict, f'Image name {image_name} not found in detections:\n{detections_dict.keys()}'
    rects, confs = detections_dict[image_name]
    detections = sorted([(r, c) for r, c in zip(rects, confs) if c >= 0.7],
                        key=lambda x: x[1],
                        reverse=True)
    print(f'{image_name} found in detections')
    plt.imshow(plt.imread(str(p)))
    for rect, conf in detections:
        draw_rectangle_confidence(rect, conf)
    plt.show()