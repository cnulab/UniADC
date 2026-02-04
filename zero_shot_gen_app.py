import argparse
import gradio as gr
import numpy as np
import torch
from birefnet import BiRefNet
from torchvision import transforms
from PIL import Image
import alpha_clip
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, DDIMScheduler
from utils.mask import ZeroShotMaskGen
from utils.helper import ImageWithMask, set_seed
import os


class UniADCController:

    def __init__(self,
                 ckpt_root,
                 image_size):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

        clip_model, clip_image_preprocess = alpha_clip.load("ViT-L/14@336px",
                                                            alpha_vision_ckpt_pth=os.path.join(ckpt_root, 'clip_l14_336_grit_20m_4xe.pth'),
                                                            device=self.device)

        self.clip_model = clip_model.float()
        self.clip_image_preprocess = clip_image_preprocess
        self.clip_mask_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((336, 336)),
            transforms.Normalize(0.5, 0.26)
        ])

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
        self.birefnet.eval()

        self.birefnet_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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

    @torch.no_grad()
    def get_clip_match_score(self, images, masks, text):
        image = torch.cat([self.clip_image_preprocess(image).unsqueeze(0).to(self.device) for image in images], dim=0)
        alpha = torch.cat([self.clip_mask_preprocess((mask * 255).astype(np.uint8)).unsqueeze(0).to(self.device) for mask in masks], dim=0)
        text = alpha_clip.tokenize([text]).to(self.device)
        image_features = self.clip_model.visual(image, alpha)
        text_features = self.clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).cpu().view(-1).numpy().tolist()
        return similarity

    def check_mask(self, mask):
        return not (np.max(np.array(mask)) == 0)

    def infer(self,
              input_image,
              anomaly_prompt,
              steps,
              conditioning_scale,
              prompt_guidance_scale,
              noise_factor,
              batch_size,
              mask_types, mask_sizes,
              seed,
              is_birefnet):

        image = np.array(input_image["image"].convert("RGB").resize((self.image_size, self.image_size)))
        normal_image = ImageWithMask(image.astype(float), resolution = self.image_size)

        set_seed(seed)
        generator = torch.Generator("cuda").manual_seed(seed)

        if not self.check_mask(input_image['mask']):
            if is_birefnet:
                fg = self.segment_foreground(normal_image)
            else:
                fg = None
            mask_config = ["{}_{}".format(mask_type, mask_size) for mask_size in mask_sizes for mask_type in mask_types]
            self.maskgen = ZeroShotMaskGen(mask_config if len(mask_config) != 0 else None)

            masks = self.maskgen.generate(fg, size=self.image_size, num_mask = batch_size)
        else:
            mask = np.array(input_image["mask"].convert("L").resize((self.image_size, self.image_size)))
            mask[mask != 0] = 1
            masks = [mask] * batch_size

        image = normal_image.toArray()

        anomaly_images, _ = self.pipe.zero_shot_anomaly_syn(
            image,
            masks,
            prompt=anomaly_prompt,
            num_inference_steps=steps,
            generator=generator,
            prompt_guidance_scale=prompt_guidance_scale,
            noise_factors=noise_factor,
            conditioning_scale=conditioning_scale,
        )
        match_scores = self.get_clip_match_score(anomaly_images, masks, anomaly_prompt)

        anomaly_image = anomaly_images[np.argmax(match_scores)]
        mask = masks[np.argmax(match_scores)]
        mask = Image.fromarray(mask[..., None].astype(np.uint8).repeat(3, -1) * 255).convert("RGB")
        return [anomaly_image, mask]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Prior Guided Controllable Inpainting App")
    parser.add_argument("--ckpt_root", default="ckpt", type=str)
    parser.add_argument("--conditioning_scale", default=1.0, type=float)
    parser.add_argument("--prompt_guidance_scale", default=12.5, type=float)
    parser.add_argument("--num_inference_steps", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--noise_factor", default=0.5, type=float)
    parser.add_argument("--share", action="store_true", default=True)

    args = parser.parse_args()

    uniadc = UniADCController(ckpt_root = args.ckpt_root,
                              image_size = 512 )

    with gr.Blocks(css="style.css") as demo:
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='18'>UniADC zero-shot Anomaly Synthesis</font></div>"  # noqa
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input image and draw mask")
                input_image = gr.Image(source="upload", tool="sketch", type="pil")
                with gr.Row():
                    anomaly_prompt = gr.Textbox(label="Anomaly Description")

                run_button = gr.Button(label="Run")

                with gr.Accordion("Advanced options", open=False):
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=args.num_inference_steps, step=1)

                    mask_types = gr.CheckboxGroup(
                        choices=["Rectangle", "Line", "Polygon", "Perlin", "Brush", "Ellipse", "HollowEllipse"],
                        value=["Rectangle", "Line", "Polygon", "Perlin", "Brush", "Ellipse", "HollowEllipse"],
                        label="Mask Types",
                    )

                    mask_sizes = gr.CheckboxGroup(
                        choices=["Large", "Medium", "Small"],
                        value=["Large", "Medium", "Small"],
                        label="Mask Size",
                    )

                    noise_factor = gr.Slider(
                        label="Noise factor",
                        minimum=0.1,
                        maximum=1.0,
                        value=args.noise_factor,
                        step=0.1,
                    )

                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=24,
                        step=1,
                        value=args.batch_size,
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
                mask_types, mask_sizes,
                seed,
                is_birefnet
            ],
            outputs=[inpaint_result],
        )
        demo.queue()
        demo.launch(share=args.share, server_name="0.0.0.0", server_port=6006)