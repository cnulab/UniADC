import argparse
import os.path
import gradio as gr
import numpy as np
import torch
from birefnet import BiRefNet
from torchvision import transforms
from PIL import Image
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, DDIMScheduler
from utils.helper import ImageWithMask, set_seed
from utils.mask import FewShotMaskGen
from skimage.metrics import structural_similarity as ssim
import os


class UniADCController:

    def __init__(self,
                 ckpt_root,
                 image_size):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

        self.brushnet = BrushNetModel.from_pretrained(os.path.join(ckpt_root, "segmentation_mask_brushnet_ckpt"), torch_dtype=self.dtype)

        self.pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            os.path.join(ckpt_root, 'realisticVisionV60B1_v51VAE'),
            brushnet=self.brushnet,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

        self.birefnet = BiRefNet.from_pretrained(os.path.join(ckpt_root, 'birefnet')).to(self.device)

        self.birefnet_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.birefnet.eval()
        self.image_size = image_size

    @torch.no_grad()
    def segment_foreground(self, image, min_fg_area = 0.05, fg_threshold = 0.5):
        if image.mask is None:
            birefnet_image = self.birefnet_transform(image.toImage()).unsqueeze(0).to(self.device)
        else:
            birefnet_image = self.birefnet_transform(image.toImage()[0]).unsqueeze(0).to(self.device)
        self.birefnet.eval()
        fg = self.birefnet(birefnet_image)[-1].sigmoid().cpu()
        fg[fg >= fg_threshold] = 1
        fg[fg < fg_threshold] = 0
        fg = fg.view(image.get_shape()).numpy()
        return np.ones_like(fg) if np.mean(fg) < min_fg_area else fg


    def load_anomaly_images_and_masks(self,
                                      upload_files,
                                      anomaly_size_threshold,
                                      content_ratio,
                                      anomaly_number = 1
                                      ):

        if upload_files is None:
            return []

        mask_paths = [ file.name for file in upload_files if file.name.find('mask')!=-1]
        image_paths = [ file.name for file in upload_files if file.name not in mask_paths]

        mask_names = [os.path.basename(mask_path) for mask_path in mask_paths]
        image_names = [os.path.basename(image_path) for image_path in image_paths]

        gallery_images = []
        anomaly_samples = []

        for i, image_name in enumerate(image_names):
            mask_name = image_name.replace('.', '_mask.')
            index = mask_names.index(mask_name)
            assert index != -1
            anomaly_image_and_mask = ImageWithMask(image_paths[i], mask_paths[index], resolution=self.image_size)
            anomaly_samples.append(anomaly_image_and_mask)
            image, mask = anomaly_image_and_mask.toImage()
            gallery_images.append(image)
            gallery_images.append(mask)

        self.maskgen = FewShotMaskGen(anomaly_samples,
                                 maximum_number = anomaly_number,
                                 content_ratio = content_ratio,
                                 anomaly_size_threshold = anomaly_size_threshold)
        return gallery_images


    def get_ssim_scores(self, src_images, ref_images):
        scores = []
        for src_image, ref_image in zip(src_images, ref_images):
            if not isinstance(src_image, np.ndarray):
                src_image = np.array(src_image)
            if not isinstance(ref_image, np.ndarray):
                ref_image = np.array(ref_image)
            _, diff = ssim(src_image, ref_image, channel_axis=-1, data_range=255, full=True)
            diff = np.mean(diff, axis=-1)
            score = np.mean(diff[ref_image[:, :, 0] != 0])
            scores.append(score)
        return scores


    def infer(self,
              input_image,
              anomaly_prompt,
              steps,
              conditioning_scale,
              prompt_guidance_scale,
              noise_factor,
              batch_size,
              seed,
              is_birefnet):

        image = np.array(input_image.convert("RGB").resize((self.image_size, self.image_size)))
        normal_image = ImageWithMask(image.astype(float), resolution = self.image_size)
        set_seed(seed)
        generator = torch.Generator("cuda").manual_seed(seed)

        if is_birefnet:
            fg = self.segment_foreground(normal_image)
        else:
            fg = None

        anomaly_regions = self.maskgen.sample_anomaly_regions()

        src_images, blended_images, ref_images, masks, image_paths = [], [], [], [], []

        for i in range(batch_size):
            blended_image, ref_image, mask = self.maskgen.generate(normal_image.toArray(), anomaly_regions, fg=fg, size=self.image_size)
            blended_images.append(blended_image)
            ref_images.append(ref_image)
            src_images.append(normal_image.toArray())
            masks.append(mask)

        anomaly_images, _ = self.pipe.few_shot_anomaly_syn(
            blended_images,
            masks,
            prompt=anomaly_prompt,
            num_inference_steps=steps,
            generator=generator,
            prompt_guidance_scale=prompt_guidance_scale,
            noise_factors=noise_factor,
            conditioning_scale=conditioning_scale,
        )
        scores = self.get_ssim_scores(anomaly_images, ref_images)
        anomaly_image = anomaly_images[np.argmax(scores)]
        mask = masks[np.argmax(scores)]
        mask = Image.fromarray(mask[..., None].astype(np.uint8).repeat(3, -1) * 255).convert("RGB")
        return [anomaly_image, mask]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Sample Guided Controllable Inpainting App")
    parser.add_argument("--ckpt_root", default="ckpt", type=str)
    parser.add_argument("--anomaly_size_threshold", default=64, type=int, help='The minimum number of abnormal pixels present in the reference abnormal image')
    parser.add_argument("--noise_factor", default=0.1, type=float)
    parser.add_argument("--content_ratio", default=0.25, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--prompt_guidance_scale", default=12.5, type=float)
    parser.add_argument("--conditioning_scale", default=1.0, type=float)
    parser.add_argument("--num_inference_steps", default=10, type=int)
    parser.add_argument("--share", action="store_true", default=True)

    args = parser.parse_args()
    uniadc = UniADCController(ckpt_root= args.ckpt_root,
                               image_size = 512)

    with gr.Blocks(css="style.css") as demo:
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='18'>UniADC few-shot Anomaly Synthesis</font></div>"  # noqa
            )

        with gr.Row():
            with gr.Column():
                files_uploader = gr.Files(label="upload reference anomaly images and masks \n(normal_image: xxx.jpg|xxx.png) \n(mask:xxx_mask.jpg|xxx_mask.png)"
                                                "\nPlease ensure that the anomalous images and their corresponding masks are uploaded in pairs, for example <000.png,000_mask.png>, <001.png,001_mask.png>", file_types=[".jpg", '.png'])
                images_gallery = gr.Gallery(label="upload images and masks", show_label=False, columns=2)

                input_image = gr.Image(source="upload", label="upload normal image", type="pil")
                anomaly_prompt = gr.Textbox(label="Anomaly Description")
                run_button = gr.Button(label="Run")

                with gr.Accordion("Advanced options", open=False):

                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=args.num_inference_steps, step=1)

                    noise_factor = gr.Slider(
                        label="Noise factor",
                        minimum=0.1,
                        maximum=1.0,
                        value=args.noise_factor,
                        step=0.1,
                    )

                    conditioning_scale = gr.Slider(
                        label="ControlNet Guidance Scale",
                        minimum=0.1,
                        maximum=1.0,
                        value= args.conditioning_scale,
                        step=0.1,
                    )

                    prompt_guidance_scale = gr.Slider(
                        label="Prompt Guidance Scale",
                        minimum=7.5,
                        maximum=20,
                        value=args.prompt_guidance_scale,
                        step=0.5,
                    )

                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=24,
                        step=1,
                        value=args.batch_size,
                    )

                    anomaly_size_threshold = gr.Slider(
                        label="Anomaly Size Threshold",
                        minimum=32,
                        maximum=1024,
                        step=32,
                        value=args.anomaly_size_threshold,
                    )

                    content_ratio = gr.Slider(
                        label="Normal Content Ratio",
                        minimum=0.05,
                        maximum=0.5,
                        step=0.05,
                        value=args.content_ratio,
                    )

                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=500,
                        step=1,
                        randomize=True,
                    )
                    is_birefnet = gr.Checkbox(label="Foreground segmentation", value=True)

                with gr.Column():
                    gr.Markdown("### Synthesis result")
                    inpaint_result = gr.Gallery(label="Generated images", show_label=False, columns=4)

        run_button.click(
            fn=uniadc.infer,
            inputs=[
                input_image,
                anomaly_prompt,
                steps,
                conditioning_scale,
                prompt_guidance_scale,
                noise_factor,
                batch_size,
                seed,
                is_birefnet
            ],
            outputs=[inpaint_result],
        )

        files_uploader.change(
                fn=uniadc.load_anomaly_images_and_masks,
                inputs=[
                    files_uploader,
                    anomaly_size_threshold,
                    content_ratio,
                ],
                outputs=[images_gallery],
        )

        demo.queue()
        demo.launch(share=args.share, server_name="0.0.0.0", server_port=6006)