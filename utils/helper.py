import logging
from datetime import datetime
import tabulate
from PIL import Image
import os
import random
import torch
import cv2
import numpy as np
import PIL
from dataclasses import dataclass


def map_func(storage, location):
    return storage.cuda()


def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log

norm_func = lambda x, eps=1e-8: (x - (m := x.min())) / (x.max() - m + eps)


def get_image_level_score(anomaly_detection_scores, T = 100):
    image_score, _ = torch.sort(
            anomaly_detection_scores.view(anomaly_detection_scores.size(0), -1).cuda(),
            dim=1,
            descending=True,
        )

    image_score = torch.mean(
            image_score[:, : T], dim=1
    )
    return image_score.cpu()



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette):
    """Get image color pallete for visualizing masks"""
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)
    return out_img


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count



class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )



def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



@dataclass
class ImageWithMask:
    image: np.ndarray # numpy.ndarray (height, width, channel)
    mask: np.ndarray = None # numpy.ndarray (height, width) \in [0, 1]

    def __init__(self, image_or_path, mask_or_path = None, resolution = 512):
        self.load_image(image_or_path, mask_or_path, resolution = resolution)

    def load_image(self, image_or_path, mask_or_path = None, resolution = 512):
        if isinstance(image_or_path, str):
            image = np.array(PIL.Image.open(image_or_path).resize((resolution, resolution)).convert("RGB")).astype(float)
            self.image = image
        else:
            self.image = image_or_path

        if mask_or_path is not None:
            if isinstance(mask_or_path, str):
                mask = np.array(PIL.Image.open(mask_or_path).resize((resolution, resolution)).convert("L"))
                mask[mask != 0] = 1
                self.mask = mask.astype(np.uint8)
            else:
                self.mask = mask_or_path
            assert self.image.shape[0] == self.mask.shape[0] and self.image.shape[1] == self.mask.shape[1]

    def toImage(self):
        image = PIL.Image.fromarray(self.image.astype(np.uint8))
        if self.mask is not None:
            mask = PIL.Image.fromarray((self.mask * 255).astype(np.uint8))
            return image, mask
        else:
            return image

    def toArray(self, mask_dim_expand = False):
        image = self.image
        if self.mask is not None:
            if mask_dim_expand:
                mask = self.mask[..., None].repeat(3, -1)
            else:
                mask = self.mask
            return image, mask
        else:
            return image

    def get_connect_areas(self, anomaly_size_threshold = None, content_ratio = 0):
        assert self.mask is not None
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.mask.astype(np.uint8), 8, cv2.CV_8U)
        sub_images_and_masks = []
        sub_image_stats = []

        for stat in stats[1:]:
            x, y, width, height, area = stat
            y_start = int(max(y - content_ratio * height, 0))
            y_end = int(min(y + height + content_ratio * height, self.get_shape()[0]))
            x_start = int(max(x - content_ratio * width, 0))
            x_end = int(min(x + width + content_ratio * width, self.get_shape()[1]))
            if anomaly_size_threshold is None or area >= anomaly_size_threshold:
                sub_image = self.image[y_start : y_end, x_start: x_end, :]
                sub_mask = self.mask[y_start : y_end, x_start: x_end]
                sub_images_and_masks.append(ImageWithMask(sub_image, sub_mask))
                sub_image_stats.append([x_start, y_start, x_end-x_start, y_end-y_start, area])
        return sub_images_and_masks, sub_image_stats

    def get_shape(self):
        return self.image.shape[:-1]