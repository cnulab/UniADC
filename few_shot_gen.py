import os
import torch
import numpy as np
from PIL import Image
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, DDIMScheduler
import json
import argparse
from birefnet import BiRefNet
from torchvision import transforms
from utils.mask import FewShotMaskGen
from utils.helper import ImageWithMask
import yaml
from easydict import EasyDict
from tqdm import tqdm
import random
from skimage.metrics import structural_similarity as ssim
import cv2


def add_sample_config(sample_info, config_path):
    sample_info = json.dumps(sample_info) + '\n'
    with open(config_path, 'a+') as f:
        f.writelines(sample_info)

def parse_path(image_path):
    image_root, image_name = os.path.split(image_path)
    image_root, image_anomaly_type = os.path.split(image_root)
    image_root, split = os.path.split(image_root)
    image_root, clsname = os.path.split(image_root)
    return clsname, split, image_anomaly_type, image_name


@torch.no_grad()
def segment_foreground(image, birefnet, birefnet_transform, device, min_fg_area=0.05, fg_threshold=0.5):
    if image.mask is None:
        birefnet_image = birefnet_transform(image.toImage()).unsqueeze(0).to(device)
    else:
        birefnet_image = birefnet_transform(image.toImage()[0]).unsqueeze(0).to(device)
    birefnet.eval()
    fg = birefnet(birefnet_image)[-1].sigmoid().cpu()
    fg[fg >= fg_threshold] = 1
    fg[fg < fg_threshold] = 0
    fg = fg.view(image.get_shape()).numpy()
    return np.ones_like(fg) if np.mean(fg) < min_fg_area else fg


def get_ssim_scores(src_images, ref_images):
    scores = []
    for src_image, ref_image in zip(src_images, ref_images):
        if not isinstance(src_image, np.ndarray):
            src_image = np.array(src_image)
        if not isinstance(ref_image, np.ndarray):
            ref_image = np.array(ref_image)
        _, diff = ssim(src_image, ref_image, channel_axis=-1, data_range=255, full = True)
        diff = np.mean(diff, axis = -1)
        score = np.mean(diff[ref_image[:, :, 0] != 0])
        scores.append(score)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Anomaly Sample Guided Controllable Inpainting")
    parser.add_argument("--ckpt_root", default="ckpt", type=str)

    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--dataset", default="mvtec", type=str)
    parser.add_argument("--sample_file", default="few_hazelnut_n2a1.jsonl", type=str)
    parser.add_argument("--anomaly_type", default='crack', type=str)

    parser.add_argument("--config_root", default="config", type=str)
    parser.add_argument("--experiment_root", default="experiment", type=str)
    parser.add_argument("--remove_old_sample", default=True, type=bool)

    parser.add_argument("--num_inference_steps", default=10, type=int)
    parser.add_argument("--resolution", default = 512, type=int)
    parser.add_argument("--gen_number", default=16, type=int)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    args.category = "_".join(args.sample_file.split('_')[1:-1])

    config_root = os.path.join(args.config_root, 'few-shot', args.dataset)
    with open(os.path.join(config_root, '{}.yaml'.format(args.category))) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    brushnet = BrushNetModel.from_pretrained(os.path.join(args.ckpt_root, "segmentation_mask_brushnet_ckpt"), torch_dtype=dtype)

    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        os.path.join(args.ckpt_root, 'realisticVisionV60B1_v51VAE'),
        brushnet=brushnet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    if not getattr(args.config, 'disableFG', False):
        birefnet = BiRefNet.from_pretrained(os.path.join(args.ckpt_root, 'birefnet')).to(device)

        birefnet_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    with open(os.path.join(args.experiment_root, args.dataset, args.category,
                           'samples', args.sample_file), "r+") as f:
        lines = [line.replace('\n', '') for line in f.readlines()]
        anomaly_samples = [sample for sample in [json.loads(line) for line in lines] if
                           sample['label_name'] == args.anomaly_type]
        normal_samples = [sample for sample in [json.loads(line) for line in lines] if int(sample['label']) == 0]

    label_idx = anomaly_samples[0]['label']

    position_reserved_list = getattr(args.config, 'position_reserved', None)
    if position_reserved_list is not None and args.anomaly_type in position_reserved_list:
        position_reserved = True
    else:
        position_reserved = False

    anomaly_samples = [ImageWithMask(os.path.join(args.data_root, args.dataset, sample['filename']),
        os.path.join(args.data_root, args.dataset, sample['maskname']), resolution = args.resolution) for sample in anomaly_samples]

    maskgen = FewShotMaskGen(anomaly_samples,
                             content_ratio=args.config.content_ratio,
                             anomaly_size_threshold = args.config.anomaly_size_threshold,
                             maximum_number=args.config.maximum_anomaly_number)

    for i in tqdm(range(len(normal_samples)), total=len(normal_samples), desc='normal image foreground segment...'):
        normal_sample = normal_samples[i]
        image_path = os.path.join(args.data_root, args.dataset, normal_sample['filename'])
        normal_image_and_mask = ImageWithMask(image_path, resolution = args.resolution)
        if not getattr(args.config, 'disableFG', False):
            fg = segment_foreground(normal_image_and_mask, birefnet, birefnet_transform, device)
        else:
            fg = None
        image = normal_image_and_mask.toArray()
        normal_samples[i] = [image, image_path, fg]

    generator = torch.Generator("cuda")
    sample_name = args.sample_file.replace('.', '_gen_{}.'.format(args.anomaly_type))
    save_config_path = os.path.join(args.experiment_root, args.dataset, args.category, 'samples', sample_name)

    if args.remove_old_sample and os.path.exists(save_config_path):
        os.remove(save_config_path)

    for n in tqdm(range(args.gen_number), total=args.gen_number, desc='{}: {} Few-shot Generating...'.format(args.category, args.anomaly_type)):
        samples = []
        anomaly_regions = maskgen.sample_anomaly_regions()

        for batch_size_index in range(args.config.candidates_batch_size):
            src_images, blended_images, ref_images, masks, image_paths = [], [], [], [], []

            for i in range(args.config.batch_size):
                src_image, image_path, fg = random.choice(normal_samples)
                blended_image, ref_image, mask = maskgen.generate(src_image, anomaly_regions, fg=fg, size=src_image.shape[0], position_reserved = position_reserved)
                blended_images.append(blended_image)
                ref_images.append(ref_image)
                src_images.append(src_image)
                masks.append(mask)
                image_paths.append(image_path)

            anomaly_images, _ = pipe.few_shot_anomaly_syn(
                blended_images,
                masks,
                prompt= "{} defect".format(args.anomaly_type.replace('_', ' ')),
                num_inference_steps= args.num_inference_steps,
                generator = generator,
                prompt_guidance_scale = args.config.prompt_guidance_scale,
                noise_factors = args.config.noise_factor,
                conditioning_scale = args.config.conditioning_scale,
                feature_masking=False,
            )

            samples.extend([[anomaly_image,
                            masks[index % args.config.batch_size],
                            src_images[index % args.config.batch_size],
                            ref_images[index % args.config.batch_size],
                            blended_images[index % args.config.batch_size],
                            image_paths[index % args.config.batch_size],
                                    args.config.noise_factor[ index // args.config.batch_size]
                             if isinstance(args.config.noise_factor, list) else args.config.noise_factor]
                            for index, anomaly_image in enumerate(anomaly_images)])

        scores = get_ssim_scores([sample[0] for sample in samples],
                                 [sample[3] for sample in samples])

        anomaly_image, mask, src_image, ref_image, blended_image, image_path, noise_factor = samples[np.argmax(scores)]

        if args.config.blended:
            anomaly_image = np.array(anomaly_image)
            mask_blurred = cv2.GaussianBlur(mask * 255, (13, 13), 0) / 255
            mask_np = 1 - (1 - mask) * (1 - mask_blurred)
            mask_np = mask_np[:, :, np.newaxis]
            anomaly_image = src_image * (1 - mask_np) + anomaly_image * mask_np
            anomaly_image = Image.fromarray(anomaly_image.astype(np.uint8))

        clsname, split, _, image_name = parse_path(image_path)

        save_root = os.path.join(args.experiment_root, args.dataset, args.category, "images",
                                 "few_shot", args.anomaly_type)

        os.makedirs(save_root, exist_ok=True)

        new_image_name = "{:03}.{}".format(n, image_name[image_name.find('.') + 1:])
        new_mask_name = "{:03}_mask.{}".format(n, image_name[image_name.find('.') + 1:])

        anomaly_image.save(os.path.join(save_root, new_image_name))
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(save_root, new_mask_name))

        sample_info = {"filename": os.path.join(save_root, new_image_name),
                       "maskname": os.path.join(save_root, new_mask_name),
                       "label": label_idx,
                       "label_name": args.anomaly_type,
                       "clsname": clsname}

        add_sample_config(sample_info, save_config_path)
