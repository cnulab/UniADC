import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
from skimage import morphology
from .helper import ImageWithMask
import torch
from typing import Union

def merge_masks(masks):
    merge_mask = np.zeros(masks[0].shape).astype(np.uint8)
    for mask in masks:
        merge_mask = (mask.astype(bool) | merge_mask.astype(bool)).astype(np.uint8)
    return merge_mask


class ForegroundIdentityMaskGen:
    def generate(self, fg: np.ndarray = None, size: int = None):
        assert fg is not None
        mask = fg.copy()
        mask[ mask != 0] = 1
        return mask.astype(np.uint8)


class RandomRectangleMaskGen:
    def __init__(self,
                 minimum_size_ratio: float = 0.1,
                 maximum_size_ratio: float = 0.2,
                 minimum_aspect_ratio: float = 0.5,
                 maximum_aspect_ratio: float = 1.0,
                 maximum_number: int = 1,
                 overlap_threshold: float = 0.02,
                 smoothing: int = 21):

        self.minimum_size_ratio = minimum_size_ratio
        self.maximum_size_ratio = maximum_size_ratio
        self.minimum_aspect_ratio = minimum_aspect_ratio
        self.maximum_aspect_ratio = maximum_aspect_ratio
        self.maximum_number = maximum_number
        self.smoothing = smoothing
        self.overlap_threshold = overlap_threshold
        assert self.overlap_threshold <= self.minimum_size_ratio


    def place(self, size,  x_start, y_start, x_end, y_end):

        area = random.uniform(self.minimum_size_ratio, self.maximum_size_ratio) * (size ** 2)
        aspect_ratio = random.uniform(self.minimum_aspect_ratio, self.maximum_aspect_ratio)
        max_len = int(math.sqrt(area / aspect_ratio))
        min_len = int(max_len * aspect_ratio)

        if random.choice([0, 1]) == 0:
            w, h = max_len, min_len
        else:
            w, h = min_len, max_len

        w = random.randint(int((x_end - x_start) / 2), x_end - x_start) if w > x_end - x_start else w
        h = random.randint(int((y_end - y_start) / 2), y_end - y_start) if h > y_end - y_start else h

        to_location_w = random.randint(x_start, x_end - w)
        to_location_h = random.randint(y_start, y_end - h)

        rectangle = np.zeros((size, size)).astype(np.uint8)
        rectangle[to_location_h:to_location_h+h, to_location_w:to_location_w+w] = 1
        angle = random.randint(0, 90)
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_rectangle = cv2.warpAffine(rectangle, M, (size, size), flags=cv2.INTER_NEAREST)
        return rotated_rectangle


    def generate(self, fg: np.ndarray = None, size: int = None):
        assert fg is not None or size is not None

        if fg is None:
            fg = np.ones((size, size)).astype(np.uint8)

        ys, xs = np.where(fg == 1)

        if len(xs) > 0 and len(ys) > 0:
            x_start, x_end = xs.min(), xs.max()
            y_start, y_end = ys.min(), ys.max()
        else:
            x_start, y_start, x_end, y_end = 0, 0, size, size

        anomalies_number = random.choice(list(range(1, self.maximum_number+1)))

        while True:
            mask = np.zeros_like(fg).astype(np.uint8)
            for _ in range(anomalies_number):
                rectangle = self.place( mask.shape[0], x_start, y_start, x_end, y_end)
                mask = (mask.astype(bool) | rectangle.astype(bool)).astype(np.uint8)
            mask = mask * fg
            if np.sum(mask) / np.sum(fg) >= self.overlap_threshold:
                break

        mask = cv2.GaussianBlur(mask, (self.smoothing, self.smoothing), 0)
        mask [mask!=0] = 1
        mask = morphology.closing(mask, morphology.footprint_rectangle((6, 6)))
        return mask.astype(np.uint8)


class RandomPolygonMaskGen:

    def __init__(self,
                 minimum_size_ratio: float = 0.1,
                 maximum_size_ratio: float = 0.2,
                 minimum_aspect_ratio: float = 0.5,
                 maximum_aspect_ratio: float = 1.0,
                 min_edge = 10,
                 max_edge = 20,
                 maximum_number: int = 1,
                 smoothing: int = 15,
                 overlap_threshold = 0.02):

        self.minimum_size_ratio = minimum_size_ratio
        self.maximum_size_ratio = maximum_size_ratio
        self.minimum_aspect_ratio = minimum_aspect_ratio
        self.maximum_aspect_ratio = maximum_aspect_ratio
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.maximum_number = maximum_number
        self.smoothing = smoothing
        self.overlap_threshold = overlap_threshold
        assert self.overlap_threshold <= self.minimum_size_ratio


    def uniform_random(self, left, right, size):
        rand_nums = (right - left) * np.random.random(size) + left
        return rand_nums


    def place(self, size, x_start, y_start, x_end, y_end):

        area = random.uniform(self.minimum_size_ratio, self.maximum_size_ratio) * (size ** 2)
        aspect_ratio = random.uniform(self.minimum_aspect_ratio, self.maximum_aspect_ratio)
        max_len = int(math.sqrt(area / aspect_ratio))
        min_len = int(max_len * aspect_ratio)

        if random.choice([0, 1]) == 0:
            w, h = max_len, min_len
        else:
            w, h = min_len, max_len

        w = random.randint(int((x_end - x_start) / 2), x_end - x_start) if w > x_end - x_start else w
        h = random.randint(int((y_end - y_start) / 2), y_end - y_start) if h > y_end - y_start else h

        to_location_w = random.randint(x_start, x_end - w)
        to_location_h = random.randint(y_start, y_end - h)

        edge_num = random.randint(self.min_edge, self.max_edge+1)

        angles = self.uniform_random(0, 2 * np.pi, edge_num)
        angles = np.sort(angles)

        random_radius = self.uniform_random(w, h, edge_num)
        x = np.cos(angles) * random_radius
        y = np.sin(angles) * random_radius
        x = np.expand_dims(x, 1)
        y = np.expand_dims(y, 1)

        points = np.concatenate([x, y], axis=1)
        points += np.array((to_location_w, to_location_h))
        points = np.round(points).astype(np.int32)
        polygon = np.zeros((size, size), dtype=np.uint8)
        polygon = cv2.fillPoly(polygon, [points], [1])
        return polygon


    def generate(self, fg: np.ndarray = None, size: int = None):
        assert fg is not None or size is not None

        if fg is None:
            fg = np.ones((size, size)).astype(np.uint8)

        ys, xs = np.where(fg == 1)

        if len(xs) > 0 and len(ys) > 0:
            x_start, x_end = xs.min(), xs.max()
            y_start, y_end = ys.min(), ys.max()
        else:
            x_start, y_start, x_end, y_end = 0, 0, size, size

        anomalies_number = random.choice(list(range(1, self.maximum_number + 1)))
        while True:
            mask = np.zeros_like(fg).astype(np.uint8)
            for _ in range(anomalies_number):
                polygon = self.place(mask.shape[0], x_start, y_start, x_end, y_end)
                mask = (mask.astype(bool) | polygon.astype(bool)).astype(np.uint8)
            mask = mask * fg
            if np.sum(mask) / np.sum(fg) >= self.overlap_threshold:
                break

        mask = cv2.GaussianBlur(mask, (self.smoothing, self.smoothing), 0)
        mask[mask != 0] = 1
        mask = morphology.closing(mask, morphology.footprint_rectangle((6, 6)))
        return mask.astype(np.uint8)




class RandomEllipseMaskGen:
    def __init__(self,
                 minimum_size_ratio: float = 0.1,
                 maximum_size_ratio: float = 0.2,
                 minimum_aspect_ratio: float = 0.5,
                 maximum_aspect_ratio: float = 1.0,
                 maximum_number: int = 1,
                 thickness: Union[int, list] = -1,
                 overlap_threshold: float = 0.02):

        self.minimum_size_ratio = minimum_size_ratio
        self.maximum_size_ratio = maximum_size_ratio
        self.maximum_number = maximum_number
        self.minimum_aspect_ratio = minimum_aspect_ratio
        self.maximum_aspect_ratio = maximum_aspect_ratio
        self.overlap_threshold = overlap_threshold
        self.thickness = thickness
        assert self.overlap_threshold <= self.minimum_size_ratio


    def place(self, size, x_start, y_start, x_end, y_end):

        area = random.uniform(self.minimum_size_ratio, self.maximum_size_ratio) * (size ** 2)
        aspect_ratio = random.uniform(self.minimum_aspect_ratio, self.maximum_aspect_ratio)
        max_len = int(math.sqrt(area / aspect_ratio))
        min_len = int(max_len * aspect_ratio)

        if random.choice([0, 1]) == 0:
            w, h = max_len, min_len
        else:
            w, h = min_len, max_len

        w = random.randint(int((x_end - x_start) / 2), x_end - x_start) if w > x_end - x_start else w
        h = random.randint(int((y_end - y_start) / 2), y_end - y_start) if h > y_end - y_start else h

        to_location_w = random.randint(x_start, x_end - w)
        to_location_h = random.randint(y_start, y_end - h)

        ellipse = np.zeros((size, size), dtype=np.uint8)
        if isinstance(self.thickness, list):
            thickness = random.randint(*self.thickness)
        else:
            thickness = self.thickness

        cv2.ellipse(ellipse, (to_location_w, to_location_h), (w, h),
                    random.randint(0, 360), 0, 360, [1], thickness)
        return ellipse


    def generate(self, fg: np.ndarray = None, size: int = None):
        assert fg is not None or size is not None
        if fg is None:
            fg = np.ones((size, size)).astype(np.uint8)

        ys, xs = np.where(fg == 1)

        if len(xs) > 0 and len(ys) > 0:
            x_start, x_end = xs.min(), xs.max()
            y_start, y_end = ys.min(), ys.max()
        else:
            x_start, y_start, x_end, y_end = 0, 0, size, size

        anomalies_number = random.choice(list(range(1, self.maximum_number + 1)))
        while True:
            mask = np.zeros_like(fg).astype(np.uint8)
            for _ in range(anomalies_number):
                ellipse = self.place(mask.shape[0], x_start, y_start, x_end, y_end)
                mask = (mask.astype(bool) | ellipse.astype(bool)).astype(np.uint8)
            mask = mask * fg
            if np.sum(mask) / np.sum(fg) >= self.overlap_threshold:
                break

        mask[mask != 0] = 1
        mask = morphology.closing(mask, morphology.footprint_rectangle((6, 6)))
        return mask.astype(np.uint8)



class PerlinMaskGen:
    def __init__(self,
                 min_perlin_scale = 0,
                 perlin_scale = 3,
                 overlap_threshold = 0.02,
                 perlin_noise_threshold = 0.5):

        self.min_perlin_scale = min_perlin_scale
        self.perlin_scale = perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def lerp_np(x, y, w):
        fin_out = (y - x) * w + x
        return fin_out

    @staticmethod
    def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

        tile_grads = lambda slice1, slice2: np.repeat(
            np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
        dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                         axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * PerlinMaskGen.lerp_np(PerlinMaskGen.lerp_np(n00, n10, t[..., 0]),
                                                    PerlinMaskGen.lerp_np(n01, n11, t[..., 0]), t[..., 1])

    def rotate_image_cv2(self, image, angle_range=(-90, 90)):
        angle = random.uniform(*angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        return rotated

    def get_perlin_noise(self, size):

        perlin_scalex = 2 ** random.randint(self.min_perlin_scale, self.perlin_scale)
        perlin_scaley = 2 ** random.randint(self.min_perlin_scale, self.perlin_scale)

        perlin_noise = PerlinMaskGen.rand_perlin_2d_np((size, size), (perlin_scalex, perlin_scaley))

        perlin_noise = self.rotate_image_cv2(perlin_noise, angle_range=(-90, 90))

        mask_noise = np.where(
                perlin_noise > self.perlin_noise_threshold,
                np.ones_like(perlin_noise),
                np.zeros_like(perlin_noise)
            )
        return mask_noise


    def generate(self, fg: np.ndarray = None, size: int = None):
        assert fg is not None or size is not None

        if fg is None:
            fg = np.ones((size, size)).astype(np.uint8)

        while True:
            mask = self.get_perlin_noise(fg.shape[0])
            mask = mask * fg
            if np.sum(mask) / np.sum(fg) >= self.overlap_threshold:
                break

        mask [mask!=0] = 1
        mask = morphology.closing(mask, morphology.footprint_rectangle((6, 6)))
        return mask.astype(np.uint8)


class RandomBrushMaskGen:
    def __init__(self,
                 min_width = 4,
                 max_width = 12,
                 min_num_vertex = 2,
                 max_num_vertex = 10,
                 max_tries = 4,
                 overlap_threshold = 0.02):

        self.overlap_threshold = overlap_threshold
        self.min_width = min_width
        self.max_width =  max_width
        self.min_num_vertex = min_num_vertex
        self.max_num_vertex = max_num_vertex
        self.max_tries = max_tries

    def randomBrush(self, size):
        mean_angle = 2 * math.pi / 5,
        angle_range = 2 * math.pi / 15,
        average_radius = math.sqrt(2 * size ** 2) / 8
        mask = Image.new('L', (size, size), 0)

        for _ in range(np.random.randint(self.max_tries)):
            num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []

            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))

            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)

                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i][0]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i][0]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(self.min_width, self.max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_TOP_BOTTOM)

        mask = np.asarray(mask, np.uint8)
        return mask


    def generate(self, fg: np.ndarray = None, size: int = None):
        assert fg is not None or size is not None

        if fg is None:
            fg = np.ones((size, size)).astype(np.uint8)

        while True:
            mask = self.randomBrush(fg.shape[0])
            mask = mask * fg
            if np.sum(mask) / np.sum(fg) >= self.overlap_threshold:
                break

        mask[mask != 0] = 1
        mask = morphology.closing(mask, morphology.footprint_rectangle((6, 6)))
        return mask.astype(np.uint8)



GAP_Lib = {
    'FGI': ForegroundIdentityMaskGen(),

    'Rectangle_Large': RandomRectangleMaskGen( minimum_size_ratio = 0.05,
                                               maximum_size_ratio = 0.1,
                                               minimum_aspect_ratio= 0.3,
                                               maximum_aspect_ratio= 1.0,
                                               smoothing = 21,
                                               maximum_number = 1,
                                               overlap_threshold=0.03),

    'Rectangle_Medium': RandomRectangleMaskGen(minimum_size_ratio=0.01,
                                               maximum_size_ratio=0.04,
                                               minimum_aspect_ratio=0.3,
                                               maximum_aspect_ratio=1.0,
                                               smoothing=15,
                                               maximum_number=1,
                                               overlap_threshold=0.005),

    'Rectangle_Small': RandomRectangleMaskGen(minimum_size_ratio=0.005,
                                               maximum_size_ratio=0.01,
                                               minimum_aspect_ratio=0.3,
                                               maximum_aspect_ratio=1.0,
                                               smoothing=7,
                                               maximum_number=2,
                                               overlap_threshold=0.001),

    'Line_Large': RandomRectangleMaskGen(minimum_size_ratio=0.01,
                                         maximum_size_ratio=0.02,
                                         minimum_aspect_ratio=0.05,
                                         maximum_aspect_ratio=0.1,
                                         smoothing=7,
                                         maximum_number=1,
                                         overlap_threshold=0.005),

    'Line_Medium': RandomRectangleMaskGen(minimum_size_ratio=0.005,
                                         maximum_size_ratio=0.01,
                                         minimum_aspect_ratio=0.02,
                                         maximum_aspect_ratio=0.05,
                                         smoothing=5,
                                         maximum_number=1,
                                         overlap_threshold=0.002),

    'Line_Small': RandomRectangleMaskGen( minimum_size_ratio=0.002,
                                          maximum_size_ratio=0.005,
                                          minimum_aspect_ratio=0.01,
                                          maximum_aspect_ratio=0.02,
                                          smoothing=3,
                                          maximum_number=2,
                                          overlap_threshold=0.001),

    'Line_Long': RandomRectangleMaskGen( minimum_size_ratio=0.08,
                                         maximum_size_ratio=0.15,
                                         minimum_aspect_ratio=0.0005,
                                         maximum_aspect_ratio=0.002,
                                         smoothing=3,
                                         maximum_number=2,
                                         overlap_threshold=0.001),

    'Polygon_Large': RandomPolygonMaskGen(
                                         minimum_size_ratio = 0.01,
                                         maximum_size_ratio = 0.03,
                                         minimum_aspect_ratio= 0.3,
                                         maximum_aspect_ratio= 1.0,
                                         min_edge = 10,
                                         max_edge = 20,
                                         maximum_number = 1,
                                         smoothing = 21,
                                         overlap_threshold = 0.008),

    'Polygon_Medium': RandomPolygonMaskGen(
                                            minimum_size_ratio=0.005,
                                            maximum_size_ratio=0.01,
                                            minimum_aspect_ratio=0.3,
                                            maximum_aspect_ratio=1.0,
                                            min_edge=10,
                                            max_edge=20,
                                            maximum_number=1,
                                            smoothing=15,
                                            overlap_threshold=0.003),

    'Polygon_Small': RandomPolygonMaskGen(
                                            minimum_size_ratio=0.002,
                                            maximum_size_ratio=0.005,
                                            minimum_aspect_ratio=0.3,
                                            maximum_aspect_ratio=1.0,
                                            min_edge=10,
                                            max_edge=20,
                                            maximum_number=2,
                                            smoothing=7,
                                            overlap_threshold=0.001),

    'Perlin_Large': PerlinMaskGen(
                                            min_perlin_scale = 0,
                                            perlin_scale = 1,
                                            overlap_threshold = 0.02),
    'Perlin_Medium': PerlinMaskGen(
                                            min_perlin_scale = 2,
                                            perlin_scale = 3,
                                            overlap_threshold = 0.01),

    'Perlin_Small': PerlinMaskGen(
                                            min_perlin_scale = 4,
                                            perlin_scale = 5,
                                            overlap_threshold = 0.005),

    'Brush_Large': RandomBrushMaskGen(
                                        min_width = 18,
                                        max_width = 24,
                                        min_num_vertex = 2,
                                        max_num_vertex = 10,
                                        max_tries = 2,
                                        overlap_threshold = 0.01),

    'Brush_Medium': RandomBrushMaskGen(
                                        min_width = 12,
                                        max_width = 18,
                                        min_num_vertex = 2,
                                        max_num_vertex = 10,
                                        max_tries = 4,
                                        overlap_threshold = 0.005),

    'Brush_Small': RandomBrushMaskGen(
                                    min_width = 6,
                                    max_width = 12,
                                    min_num_vertex = 2,
                                    max_num_vertex = 10,
                                    max_tries = 4,
                                    overlap_threshold = 0.001),

    'Brush_Tiny': RandomBrushMaskGen(
                                    min_width=4,
                                    max_width=8,
                                    min_num_vertex=1,
                                    max_num_vertex=3,
                                    max_tries=2,
                                    overlap_threshold=0.0001),

    'NearCircularEllipse_Large': RandomEllipseMaskGen(
                                    minimum_size_ratio = 0.015,
                                    maximum_size_ratio = 0.03,
                                    minimum_aspect_ratio = 0.5,
                                    maximum_aspect_ratio= 1.0,
                                    maximum_number = 1,
                                    overlap_threshold = 0.01),

    'NearCircularEllipse_Medium': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.005,
                                        maximum_size_ratio=0.015,
                                        minimum_aspect_ratio=0.5,
                                        maximum_aspect_ratio=1.0,
                                        maximum_number=1,
                                        overlap_threshold=0.003 ),

    'NearCircularEllipse_Small': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.0002,
                                        maximum_size_ratio=0.001,
                                        minimum_aspect_ratio=0.5,
                                        maximum_aspect_ratio=1.0,
                                        maximum_number=2,
                                        overlap_threshold=0.0001),


    'EccentricEllipse_Large': RandomEllipseMaskGen(
                                        minimum_size_ratio = 0.01,
                                        maximum_size_ratio = 0.02,
                                        minimum_aspect_ratio = 0.1,
                                        maximum_aspect_ratio = 0.25,
                                        maximum_number=1,
                                        overlap_threshold = 0.005),

    'EccentricEllipse_Medium': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.005,
                                        maximum_size_ratio=0.008,
                                        minimum_aspect_ratio=0.1,
                                        maximum_aspect_ratio=0.25,
                                        maximum_number=1,
                                        overlap_threshold=0.003),

    'EccentricEllipse_Small': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.002,
                                        maximum_size_ratio=0.005,
                                        minimum_aspect_ratio=0.1,
                                        maximum_aspect_ratio=0.25,
                                        maximum_number=2,
                                        overlap_threshold=0.001),

    'NearCircularHollowEllipse_Large': RandomEllipseMaskGen(
                                        minimum_size_ratio = 0.015,
                                        maximum_size_ratio = 0.03,
                                        minimum_aspect_ratio = 0.5,
                                        maximum_aspect_ratio= 1.0,
                                        thickness= [12, 24],
                                        maximum_number = 1,
                                        overlap_threshold = 0.0075),


    'NearCircularHollowEllipse_Medium': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.005,
                                        maximum_size_ratio=0.015,
                                        minimum_aspect_ratio=0.5,
                                        maximum_aspect_ratio=1.0,
                                        thickness=[8, 16],
                                        maximum_number=1,
                                        overlap_threshold=0.0025),

    'NearCircularHollowEllipse_Small': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.001,
                                        maximum_size_ratio=0.005,
                                        minimum_aspect_ratio=0.5,
                                        maximum_aspect_ratio=1.0,
                                        thickness=[4, 10],
                                        maximum_number=2,
                                        overlap_threshold=0.0005),

    'EccentricHollowEllipse_Large': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.01,
                                        maximum_size_ratio=0.02,
                                        minimum_aspect_ratio=0.1,
                                        maximum_aspect_ratio=0.25,
                                        thickness=[12, 20],
                                        maximum_number=1,
                                        overlap_threshold=0.005),

    'EccentricHollowEllipse_Medium': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.005,
                                        maximum_size_ratio=0.008,
                                        minimum_aspect_ratio=0.1,
                                        maximum_aspect_ratio=0.25,
                                        thickness=[8, 12],
                                        maximum_number=1,
                                        overlap_threshold=0.0025),

    'EccentricHollowEllipse_Small': RandomEllipseMaskGen(
                                        minimum_size_ratio=0.002,
                                        maximum_size_ratio=0.005,
                                        minimum_aspect_ratio=0.1,
                                        maximum_aspect_ratio=0.25,
                                        thickness=[4, 8],
                                        maximum_number=2,
                                        overlap_threshold=0.0005),
}



class ZeroShotMaskGen:

    def __init__(self, mask_config = None):
        if mask_config is None:
            mask_names = [mask_name for mask_name in GAP_Lib if not mask_name.startswith('FG')]
        elif isinstance(mask_config, str):
            mask_names = [mask_name for mask_name in GAP_Lib if mask_name.find(mask_config) != -1]
        elif isinstance(mask_config, list):
            mask_names = []
            for item in mask_config:
                mask_names.extend(mask_name for mask_name in GAP_Lib if mask_name.find(item) != -1)
        else:
            raise NotImplementedError('Unsupported mask configuration.')

        mask_names = list(set(mask_names))
        print('Loading Mask Names: {}'.format(mask_names))
        self.generators = [GAP_Lib[mask_name] for mask_name in mask_names]

    def generate(self,
                 fg: np.ndarray = None,
                 size: int = None,
                 num_mask: int = None,
                 num_channel = None):

        if num_mask is None:
            num_mask = 1

        masks = [random.choice(self.generators).generate(fg, size) for _ in range(num_mask)]

        if num_channel is None:
            return masks
        else:
            return [mask[..., None].repeat(num_channel, -1) for mask in masks]



class FewShotMaskGen:
    def __init__(self,
                 anomaly_samples,
                 content_ratio: int = 0,
                 anomaly_size_threshold: int = None,
                 maximum_number: int = 1):

        self.anomaly_regions = []
        self.anomaly_stats = []

        self.content_ratio = content_ratio
        self.anomaly_size_threshold = anomaly_size_threshold

        for anomaly_sample in anomaly_samples:
            anomaly_region, anomaly_stat = anomaly_sample.get_connect_areas(anomaly_size_threshold = self.anomaly_size_threshold,
                                                                                          content_ratio = self.content_ratio)
            self.anomaly_regions.extend(anomaly_region)
            self.anomaly_stats.extend(anomaly_stat)

        self.maximum_number = maximum_number
        self.overlap_threshold = (min([stat[-1] for stat in self.anomaly_stats]) / np.prod(anomaly_samples[0].get_shape())) * 0.5

    def anomaly_source_aug(self, dst_resion: ImageWithMask):
        aug_result = ImageWithMask(dst_resion.image.copy(), dst_resion.mask.copy())
        is_transpose = random.choice([False, True])
        if is_transpose:
            aug_result.image = aug_result.image.transpose((1, 0, 2))
            aug_result.mask = aug_result.mask.T
        mirroring = random.choice(['none', 'flip', 'vertical', 'rotate'])
        if mirroring == 'flip':
            aug_result.image = np.ascontiguousarray(aug_result.image[:, ::-1, :])
            aug_result.mask = aug_result.mask[:, ::-1]
        elif mirroring == 'vertical':
            aug_result.image = np.ascontiguousarray(aug_result.image[::-1, :, :])
            aug_result.mask = aug_result.mask[::-1, :]
        elif mirroring == 'rotate':
            image = aug_result.image
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, random.randint(0, 90), 1.0)
            rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            rotated_mask = cv2.warpAffine(aug_result.mask, M, (aug_result.mask.shape[1], aug_result.mask.shape[0]))
            aug_result = ImageWithMask(rotated_image, rotated_mask)
        return aug_result


    def sample_anomaly_regions(self):
        anomalies_number = random.choice(list(range(1, self.maximum_number + 1)))
        regions = []
        for _ in range(anomalies_number):
            index = random.choice(list(range(len(self.anomaly_regions))))
            anomaly_region = self.anomaly_regions[index]
            anomaly_state = self.anomaly_stats[index]
            regions.append([anomaly_region, anomaly_state])
        return regions


    def place(self, src_image, ref_image, anomaly_region, anomaly_state, fg, x_start, y_start, x_end, y_end, position_reserved):

        if position_reserved:
            mask = np.zeros_like(fg).astype(np.uint8)
            x, y, width, height, area = anomaly_state
            mask[y : y + height, x: x + width] = anomaly_region.mask
            mask = (mask * fg).astype(np.uint8)
            anomaly_sub_image = anomaly_region.toArray()[0]
            src_image[y : y + height, x: x + width, :] = src_image[y : y + height, x: x + width, :] * (1 - mask[y : y + height, x: x + width][..., None]) + \
                                                         anomaly_sub_image * mask[y : y + height, x: x + width][..., None]

            ref_image[y: y + height, x: x + width, :] = anomaly_sub_image
        else:
            anomaly_region = self.anomaly_source_aug(anomaly_region)
            h, w = anomaly_region.get_shape()
            if x_end-x_start < w or y_end-y_start < h:
                to_location_w = random.randint(0, fg.shape[0]-w)
                to_location_h = random.randint(0, fg.shape[0]-h)
            else:
                to_location_w = random.randint(x_start, max(x_start+1, x_end - w))
                to_location_h = random.randint(y_start, max(y_start+1, y_end - h))
            mask = np.zeros_like(fg).astype(np.uint8)
            mask[to_location_h: to_location_h + h, to_location_w: to_location_w + w] = anomaly_region.mask
            mask = (mask * fg).astype(np.uint8)
            anomaly_sub_image = anomaly_region.toArray()[0]
            src_image[to_location_h:to_location_h + h, to_location_w:to_location_w + w, :] = src_image[to_location_h:to_location_h + h, to_location_w:to_location_w + w, :] * (
                     1 - mask[to_location_h:to_location_h + h, to_location_w:to_location_w + w][..., None]) + \
                    anomaly_sub_image * mask[to_location_h:to_location_h + h, to_location_w:to_location_w + w][..., None]
            ref_image[to_location_h:to_location_h + h, to_location_w:to_location_w + w, :] = anomaly_sub_image
        return src_image, ref_image, mask


    def generate(self, src_image, anomaly_regions, fg: np.ndarray = None, size: int = None, position_reserved = False):
        assert fg is not None or size is not None
        if fg is None:
            fg = np.ones((size, size)).astype(np.uint8)
        ys, xs = np.where(fg == 1)

        if len(xs) > 0 and len(ys) > 0:
            x_start, x_end = xs.min(), xs.max()
            y_start, y_end = ys.min(), ys.max()
        else:
            x_start, y_start, x_end, y_end = 0, 0, size, size

        while True:
            image = src_image.copy()
            ref_image = np.zeros_like(image)
            masks = []
            for anomaly_region, anomaly_state in anomaly_regions:
                image, ref_image, mask = self.place(image, ref_image, anomaly_region, anomaly_state, fg, x_start, y_start, x_end, y_end, position_reserved)
                masks.append(mask)
            mask = merge_masks(masks)
            mask = morphology.closing(mask, morphology.footprint_rectangle((6, 6)))
            if np.sum(mask) / np.sum(fg) >= self.overlap_threshold:
                return image, ref_image, mask
