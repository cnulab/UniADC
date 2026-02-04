import inspect
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import cv2
from utils.helper import ImageWithMask
from ..image_processor import PipelineImageInput, VaeImageProcessor
from ..loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ..models import AutoencoderKL, BrushNetModel, ImageProjection, UNet2DConditionModel
from ..lora import adjust_lora_scale_text_encoder
from ..schedulers import KarrasDiffusionSchedulers
from ..utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from ..utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from .pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from .stable_diffusion.safety_checker import StableDiffusionSafetyChecker

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
        from diffusers.utils import load_image
        import torch
        import cv2
        import numpy as np
        from PIL import Image

        base_model_path = "runwayml/stable-diffusion-v1-5"
        brushnet_path = "ckpt_path"

        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
        )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # remove following line if xformers is not installed or when using Torch 2.0.
        # pipe.enable_xformers_memory_efficient_attention()
        # memory optimization.
        pipe.enable_model_cpu_offload()

        image_path="examples/brushnet/src/test_image.jpg"
        mask_path="examples/brushnet/src/test_mask.jpg"
        caption="A cake on the table."

        init_image = cv2.imread(image_path)
        mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
        init_image = init_image * (1-mask_image)

        init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

        generator = torch.Generator("cuda").manual_seed(1234)

        image = pipe(
            caption, 
            init_image, 
            mask_image, 
            num_inference_steps=50, 
            generator=generator,
            paintingnet_conditioning_scale=1.0
        ).images[0]
        image.save("output.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class StableDiffusionBrushNetPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with BrushNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        brushnet ([`BrushNetModel`]`):
            Provides additional conditioning to the `unet` during the denoising process.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            brushnet: BrushNetModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = False,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            brushnet=brushnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
            self,
            prompt,
            device,
            do_classifier_free_guidance,
            negative_prompt=None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use discriminator free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
            )

            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]

            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype

        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]

            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]

            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        if do_classifier_free_guidance:
            return prompt_embeds, negative_prompt_embeds
        else:
            return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
            self,
            prompt,
            image,
            mask,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            brushnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )

        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.brushnet, torch._dynamo.eval_frame.OptimizedModule
        )

        if (
                isinstance(self.brushnet, BrushNetModel) or is_compiled and isinstance(self.brushnet._orig_mod,
                                                                                       BrushNetModel)
        ):
            self.check_image(image, mask, prompt, prompt_embeds)
        else:
            assert False

        # Check `brushnet_conditioning_scale`
        if (
                isinstance(self.brushnet, BrushNetModel)
                or is_compiled
                and isinstance(self.brushnet._orig_mod, BrushNetModel)
        ):
            if not isinstance(brushnet_conditioning_scale, float):
                raise TypeError("For single brushnet: `brushnet_conditioning_scale` must be type `float`.")
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def check_image(self, image, mask, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        mask_is_pil = isinstance(mask, PIL.Image.Image)
        mask_is_tensor = isinstance(mask, torch.Tensor)
        mask_is_np = isinstance(mask, np.ndarray)
        mask_is_pil_list = isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image)
        mask_is_tensor_list = isinstance(mask, list) and isinstance(mask[0], torch.Tensor)
        mask_is_np_list = isinstance(mask, list) and isinstance(mask[0], np.ndarray)

        if (
                not mask_is_pil
                and not mask_is_tensor
                and not mask_is_np
                and not mask_is_pil_list
                and not mask_is_tensor_list
                and not mask_is_np_list
        ):
            raise TypeError(
                f"mask must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(mask)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def prepare_image(
            self,
            image,
            device=None,
            dtype=None,
    ):

        image = self.image_processor.preprocess(image).to(dtype=torch.float32)
        if device is not None:
            image = image.to(device=device)
        if dtype is not None:
            image = image.to(dtype=dtype)
        return image


    def prepare_latents(self, images, ts, num_inference_steps, noise_factors, dtype, device, generator, supp_noise = None):
        image_latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        latents = []
        noises = []
        for t, noise_factor in zip(ts, noise_factors):
            if supp_noise is None:
                noise = torch.randn(image_latents.shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.repeat_interleave(supp_noise, image_latents.shape[0], dim=0)
            self.scheduler.set_timesteps(int(num_inference_steps / noise_factor), device=device)
            latent = self.scheduler.add_noise(image_latents, noise, t) * self.scheduler.init_noise_sigma
            latents.append(latent)
            noises.extend(torch.chunk(noise, chunks=noise.shape[0]))
        return latents, noises


    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no discriminator free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def few_shot_anomaly_syn(
            self,
            images: PipelineImageInput = None,
            masks: Union[List[PipelineImageInput], PipelineImageInput] = None,
            prompt = "",
            noise = None,
            prompt_guidance_scale = 7.5,
            num_inference_steps: int = 50,
            noise_factors : Union[List[float], float] = 0.5,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            conditioning_scale: Union[float, List[float]] = 1.0,
            feature_masking = True,
            **kwargs,
    ):

        if not isinstance(images, list):
            images = [images]

        if not isinstance(masks, list):
            masks = [masks]

        if not isinstance(noise_factors, list):
            noise_factors = [noise_factors]


        assert len(images) == len(masks)

        brushnet = self.brushnet._orig_mod if is_compiled_module(self.brushnet) else self.brushnet

        device = self._execution_device
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance=True,
        )

        prompt_embeds = torch.cat(([negative_prompt_embeds] * len(images) + [prompt_embeds] * len(images)) * len(noise_factors))

        if isinstance(brushnet, BrushNetModel):

            masks = np.stack(masks)[..., None].repeat(3, -1)
            images = np.stack(images)
            brush_images = images * (1-masks) if feature_masking else images

            brush_masks = self.prepare_image(
                image = masks * 255,
                device=device,
                dtype=brushnet.dtype,
            )

            src_images = self.prepare_image(
                image = images,
                device=device,
                dtype=brushnet.dtype,
            )

            brush_images = self.prepare_image(
                image=brush_images,
                device=device,
                dtype=brushnet.dtype,
            )

            brush_masks = (brush_masks.sum(1)[:, None, :, :] < 0).to(prompt_embeds.dtype)

        else:
            assert False

        noise_factors = [max([noise_factor, 0.01]) for noise_factor in noise_factors]

        timestep_bank = [ ]
        for noise_factor in noise_factors:
            timesteps, _ = retrieve_timesteps(self.scheduler, int(num_inference_steps / noise_factor), device)
            if len(timesteps) != num_inference_steps:
                timesteps = timesteps[len(timesteps) - num_inference_steps:]
            timestep_bank.append(timesteps)

        self._num_timesteps = num_inference_steps

        latents, noises = self.prepare_latents(
            src_images,
            [timesteps[0] for timesteps in timestep_bank],
            num_inference_steps,
            noise_factors,
            prompt_embeds.dtype,
            device,
            generator,
            supp_noise=noise,
        )

        brush_conditioning_latents = self.vae.encode(brush_images).latent_dist.sample() * self.vae.config.scaling_factor

        brush_masks = torch.nn.functional.interpolate(
            brush_masks,
            size=(
                brush_conditioning_latents.shape[-2],
                brush_conditioning_latents.shape[-1]
            )
        )

        conditioning_latents = torch.concat([brush_conditioning_latents, brush_masks], 1)
        conditioning_latents = torch.cat([conditioning_latents] * 2 * len(noise_factors), 0)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        is_unet_compiled = is_compiled_module(self.unet)
        is_brushnet_compiled = is_compiled_module(self.brushnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        for i in range(len(timestep_bank[0])):

            ts = []
            for timesteps in timestep_bank:
                ts.extend([timesteps[i]] * len(src_images) * 2)
            ts = torch.stack(ts).to(device)

            if (is_unet_compiled and is_brushnet_compiled) and is_torch_higher_equal_2_1:
                torch._inductor.cudagraph_mark_step_begin()

            latent_model_input = []
            for latent in latents:
                latent_model_input.extend([latent] * 2)
            latent_model_input = torch.cat(latent_model_input, dim=0)

            control_model_input = latent_model_input
            brushnet_prompt_embeds = prompt_embeds

            down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.brushnet(
                control_model_input,
                ts,
                encoder_hidden_states=brushnet_prompt_embeds,
                brushnet_cond = conditioning_latents,
                conditioning_scale = 1.0 if feature_masking else conditioning_scale,
                return_dict = False,
            )

            noise_pred = self.unet(
                latent_model_input,
                ts,
                encoder_hidden_states=prompt_embeds,
                down_block_add_samples=down_block_res_samples,
                mid_block_add_sample=mid_block_res_sample,
                up_block_add_samples=up_block_res_samples,
                return_dict=False,
            )[0]

            for t_index, time_noise in enumerate(noise_pred.chunk(len(noise_factors))):
                noise_pred_uncond, noise_pred_prompt = time_noise.chunk(2)
                noise_pred = noise_pred_uncond + prompt_guidance_scale * (noise_pred_prompt - noise_pred_uncond)
                self.scheduler.set_timesteps(int(num_inference_steps / noise_factors[t_index]), device=device)
                latents[t_index] = self.scheduler.step(noise_pred, timestep_bank[t_index][i],
                                                       latents[t_index], **extra_step_kwargs, return_dict=False)[0]

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.brushnet.to("cpu")
            torch.cuda.empty_cache()

        images = [self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                  for latent in latents]
        images = torch.cat(images, dim=0)
        do_denormalize = [True] * images.shape[0]
        images = self.image_processor.postprocess(images, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()
        return images, noises


    @torch.no_grad()
    def zero_shot_anomaly_syn(
            self,
            image: PipelineImageInput = None,
            masks: Union[List[PipelineImageInput], PipelineImageInput] = None,
            prompt: Union[str, List[str]] = None,
            noise = None,
            prompt_guidance_scale: float = 7.5,
            num_inference_steps: int = 50,
            noise_factors: Union[List[float], float] = 0.5,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            conditioning_scale: Union[float, List[float]] = 1.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ):
        if not isinstance(masks, list):
            masks = [masks]

        if not isinstance(noise_factors, list):
            noise_factors = [noise_factors]

        brushnet = self.brushnet._orig_mod if is_compiled_module(self.brushnet) else self.brushnet

        self._guidance_scale = prompt_guidance_scale

        device = self._execution_device

        if negative_prompt is not None:
            noise_factors = [1.0]
            num_inference_steps = 20

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            self.do_classifier_free_guidance,
            negative_prompt,
        )

        prompt_embeds = torch.cat(
            ([negative_prompt_embeds] * len(masks) + [prompt_embeds] * len(masks)) * len(noise_factors))

        if isinstance(brushnet, BrushNetModel):

            images = []
            brush_images = []
            brush_masks = []

            for mask in masks:
                if mask.ndim == 2:
                    mask = mask[..., None].repeat(3, -1)
                mask[mask != 0] = 1
                images.append(image)
                brush_images.append(image * (1 - mask))
                brush_masks.append(mask * 255)

            images = np.stack(images)
            brush_masks = np.stack(brush_masks)
            brush_images = np.stack(brush_images)

            images = self.prepare_image(
                image=images,
                device=device,
                dtype=brushnet.dtype,
            )
            brush_images = self.prepare_image(
                image=brush_images,
                device=device,
                dtype=brushnet.dtype,
            )
            brush_masks = self.prepare_image(
                image=brush_masks,
                device=device,
                dtype=brushnet.dtype,
            )

            brush_masks = (brush_masks.sum(1)[:, None, :, :] < 0).to(images.dtype)
        else:
            assert False

        noise_factors = [max([noise_factor, 0.1]) for noise_factor in noise_factors]

        timestep_bank = []
        for noise_factor in noise_factors:
            timesteps, _ = retrieve_timesteps(self.scheduler, int(num_inference_steps / noise_factor), device)
            if len(timesteps) != num_inference_steps:
                timesteps = timesteps[len(timesteps) - num_inference_steps:]
            timestep_bank.append(timesteps)

        self._num_timesteps = num_inference_steps

        latents, noises = self.prepare_latents(
            images,
            [timesteps[0] for timesteps in timestep_bank],
            num_inference_steps,
            noise_factors,
            prompt_embeds.dtype,
            device,
            generator,
            supp_noise = noise,
        )

        brush_conditioning_latents = self.vae.encode(brush_images).latent_dist.sample() * self.vae.config.scaling_factor

        brush_masks = torch.nn.functional.interpolate(
            brush_masks,
            size=(
                brush_conditioning_latents.shape[-2],
                brush_conditioning_latents.shape[-1]
            )
        )
        conditioning_latents = torch.concat([brush_conditioning_latents, brush_masks], 1)
        conditioning_latents = torch.cat([conditioning_latents] * 2 * len(noise_factors), 0)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        is_unet_compiled = is_compiled_module(self.unet)
        is_brushnet_compiled = is_compiled_module(self.brushnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        for i in range(len(timestep_bank[0])):
            ts = []
            for timesteps in timestep_bank:
                ts.extend([timesteps[i]] * 2 * len(masks))
            ts = torch.stack(ts).to(device)

            if (is_unet_compiled and is_brushnet_compiled) and is_torch_higher_equal_2_1:
                torch._inductor.cudagraph_mark_step_begin()

            latent_model_input = []
            for latent in latents:
                latent_model_input.extend([latent] * 2)

            latent_model_input = torch.cat(latent_model_input, dim=0)
            control_model_input = latent_model_input
            brushnet_prompt_embeds = prompt_embeds

            down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.brushnet(
                control_model_input,
                ts,
                encoder_hidden_states=brushnet_prompt_embeds,
                brushnet_cond=conditioning_latents,
                conditioning_scale=conditioning_scale,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                ts,
                encoder_hidden_states=prompt_embeds,
                down_block_add_samples=down_block_res_samples,
                mid_block_add_sample=mid_block_res_sample,
                up_block_add_samples=up_block_res_samples,
                return_dict=False,
            )[0]

            for t_index, time_noise in enumerate(noise_pred.chunk(len(noise_factors))):
                noise_pred_uncond, noise_pred_prompt = time_noise.chunk(2)
                noise_pred = noise_pred_uncond + prompt_guidance_scale * (noise_pred_prompt - noise_pred_uncond)
                self.scheduler.set_timesteps(int(num_inference_steps / noise_factors[t_index]), device=device)
                latents[t_index] = self.scheduler.step(noise_pred, timestep_bank[t_index][i],
                                                       latents[t_index], **extra_step_kwargs, return_dict=False)[0]

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.brushnet.to("cpu")
            torch.cuda.empty_cache()

        images = [self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                  for latent in latents]
        images = torch.cat(images, dim=0)
        do_denormalize = [True] * images.shape[0]
        images = self.image_processor.postprocess(images, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()
        return images, noises
