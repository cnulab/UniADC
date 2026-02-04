import os
import torch
import numpy as np
import PIL
from PIL import Image
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, DDIMScheduler
import json
import argparse
from birefnet import BiRefNet
from torchvision import transforms
from more_itertools import chunked
import alpha_clip
from utils.mask import ZeroShotMaskGen
from utils.helper import ImageWithMask
import yaml
from easydict import EasyDict
from tqdm import tqdm
import random
import torch.nn.functional as F


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
def segment_foreground(image, birefnet, birefnet_transform, device, min_fg_area = 0.05, fg_threshold = 0.5):
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


@torch.no_grad()
def get_clip_match_score(anomaly_images,
                         masks,
                         positive_text,
                         negative_texts,
                         clip_model,
                         clip_image_preprocess,
                         clip_mask_preprocess,
                         device):

    image = torch.cat([clip_image_preprocess(image).unsqueeze(0).to(device) for image in anomaly_images], dim=0)
    alpha = torch.cat([clip_mask_preprocess((mask * 255).astype(np.uint8)).unsqueeze(0).to(device) for mask in masks], dim=0)

    image_features = clip_model.visual(image, alpha)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    text_embeddings = alpha_clip.tokenize([positive_text] + negative_texts).to(device)

    text_features = clip_model.encode_text(text_embeddings)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_similarities = (image_features @ text_features.T)
    positive_softmax_scores = F.softmax(text_similarities, dim=1)[:, 0]

    return positive_softmax_scores.cpu().numpy().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Anomaly Prior Guided Controllable Inpainting")
    parser.add_argument("--ckpt_root", default="ckpt", type=str)
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--dataset", default="mvtec", type=str)

    parser.add_argument("--sample_file", default="zero_hazelnut_n2.jsonl", type=str)
    parser.add_argument("--anomaly_type", default='crack', type=str)
    parser.add_argument("--config_root", default="config", type=str)
    parser.add_argument("--experiment_root", default="experiment", type=str)
    parser.add_argument("--remove_old_sample", default=True, type=bool)
    parser.add_argument("--num_inference_steps", default = 10, type=int)
    parser.add_argument("--resolution", default = 512, type=int)
    parser.add_argument("--gen_number", default = 16, type=int)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    args.category = "_".join(args.sample_file.split('_')[1:-1])

    config_root = os.path.join(args.config_root, 'zero-shot', args.dataset)
    with open(os.path.join(config_root, '{}.yaml'.format(args.category))) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    anomaly_type_config = getattr(args.config.anomaly_type_configs, args.anomaly_type)

    prompts_from_other_categories = []
    for key in args.config.anomaly_type_configs:
        if key != args.anomaly_type:
            prompts_from_other_categories.extend(args.config.anomaly_type_configs[key]['prompt'])

    if 'negative_prompt' not in anomaly_type_config:
        brushnet = BrushNetModel.from_pretrained(os.path.join(args.ckpt_root, "segmentation_mask_brushnet_ckpt"), torch_dtype=torch.float16)
        print('load segmentation_mask_brushnet_ckpt')
    else:
        brushnet = BrushNetModel.from_pretrained(os.path.join(args.ckpt_root, "random_mask_brushnet_ckpt"), torch_dtype=torch.float16)
        print('load random_mask_brushnet_ckpt')

    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        os.path.join(args.ckpt_root, 'realisticVisionV60B1_v51VAE'),
        brushnet = brushnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    if not getattr(anomaly_type_config, 'disableFG', False):
        birefnet = BiRefNet.from_pretrained(os.path.join(args.ckpt_root, 'birefnet')).to(device)
        birefnet_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    clip_model, clip_image_preprocess = alpha_clip.load("ViT-L/14@336px",
                                        alpha_vision_ckpt_pth=os.path.join(args.ckpt_root, 'clip_l14_336_grit_20m_4xe.pth'),
                                        device=device)
    clip_model = clip_model.float()

    clip_mask_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((336, 336)),
        transforms.Normalize(0.5, 0.26)
    ])

    with open(os.path.join(args.experiment_root, args.dataset, args.category,
                           'samples', args.sample_file), "r+") as f:
        lines = [line.replace('\n', '') for line in f.readlines()]
        normal_samples = [sample for sample in [json.loads(line) for line in lines] if int(sample['label']) == 0]

    for i in tqdm(range(len(normal_samples)), total=len(normal_samples), desc='normal image foreground segment...'):
        normal_sample = normal_samples[i]
        image_path = os.path.join(args.data_root, args.dataset, normal_sample['filename'])
        normal_image = ImageWithMask(image_path, resolution = args.resolution)
        if not getattr(anomaly_type_config, 'disableFG', False):
            fg = segment_foreground(normal_image, birefnet, birefnet_transform, device)
        else:
            fg = None
        image = normal_image.toArray()
        normal_samples[i] = [image, image_path, fg]

    maskgen = ZeroShotMaskGen(getattr(anomaly_type_config, 'masks', None))
    generator = torch.Generator("cuda")

    sample_name = args.sample_file.replace('.', '_gen_{}.'.format(args.anomaly_type))
    save_config_path = os.path.join(args.experiment_root, args.dataset, args.category, 'samples', sample_name)

    if args.remove_old_sample and os.path.exists(save_config_path):
        os.remove(save_config_path)

    for n in tqdm(range(args.gen_number), total=args.gen_number, desc='{}: {} Zero-shot Generating...'.format(args.category, args.anomaly_type)):
        samples = []

        prompt = random.choice(anomaly_type_config['prompt'])
        negative_prompt = random.choice(anomaly_type_config['negative_prompt']) if 'negative_prompt' in anomaly_type_config else None

        for batch_size_index in range(args.config.candidates_batch_size):

            src_image, image_path, fg = random.choice(normal_samples)
            masks = maskgen.generate(fg, size= src_image.shape[0], num_mask = args.config.batch_size)

            anomaly_images, _ = pipe.zero_shot_anomaly_syn(
                            src_image,
                            masks,
                            prompt = prompt,
                            negative_prompt= negative_prompt,
                            num_inference_steps = args.num_inference_steps,
                            generator = generator,
                            prompt_guidance_scale = args.config.prompt_guidance_scale,
                            noise_factors = args.config.noise_factor,
                            conditioning_scale = args.config.conditioning_scale)

            samples.extend([[anomaly_image,
                            masks[index % args.config.batch_size],
                            src_image, prompt, image_path,
                            args.config.noise_factor[index // args.config.batch_size]
                            if isinstance(args.config.noise_factor, list) else args.config.noise_factor]
                                for index, anomaly_image in enumerate(anomaly_images)])

        match_scores = []
        chunked_samples = list(chunked(samples, args.config.batch_size))

        for chunked_sample in chunked_samples:
            match_scores.extend(get_clip_match_score(
                [sample[0] for sample in chunked_sample],
                [sample[1] for sample in chunked_sample],
                prompt,
                prompts_from_other_categories,
                clip_model,
                clip_image_preprocess,
                clip_mask_preprocess,
                device))

        anomaly_image, mask, src_image, prompt, image_path, noise_factor = samples[np.argmax(match_scores)]

        clsname, split, _, image_name = parse_path(image_path)

        save_root = os.path.join(args.experiment_root, args.dataset, args.category, "images", "zero_shot", args.anomaly_type)

        os.makedirs(save_root, exist_ok=True)

        new_image_name = "{}_{:03}.{}".format(prompt.replace(' ', '_'), n, image_name[image_name.find('.') + 1:])
        new_mask_name = "{}_{:03}_mask.{}".format(prompt.replace(' ', '_'), n, image_name[image_name.find('.') + 1:])

        anomaly_image.save(os.path.join(save_root, new_image_name))

        PIL.Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(save_root, new_mask_name))

        sample_info = {"filename": os.path.join(save_root, new_image_name),
                       "maskname": os.path.join(save_root, new_mask_name),
                       "label_name": args.anomaly_type,
                       "clsname": clsname}

        add_sample_config(sample_info, save_config_path)
