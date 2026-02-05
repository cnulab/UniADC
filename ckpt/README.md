## Download Pre-trained Models
1. Request [DINOv3](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) weights, and download DINOv3-B/16 (dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth) and DINOv3-based dino.txt (dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth).  
2. Run `sh download.sh` to download other dependency weights.

All weight files are placed in this directory. The complete directory structure is as follows:
  ```
    |--ckpt                         
        |--birefnet  #BiRefNet
            |--BiRefNet_config.py
            |--birefnet.py
            |--model.safetensors
            |--...
        |--realisticVisionV60B1_v51VAE      #Stable Diffusion v1.5
            |--feature_extractor
            |--...
        |--random_mask_brushnet_ckpt        #BrushNet
            |--config.json
            |--diffusion_pytorch_model.safetensors
        |--segmentation_mask_brushnet_ckpt  #BrushNet
            |--config.json
            |--diffusion_pytorch_model.safetensors
        |--clip_l14_336_grit_20m_4xe.pth    #AlphaCLIP
        |--dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth  #DINOv3
        |--dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth #dino.txt
  
```
  
If you cannot use `download.sh` for automatic downloading, you can download manually via the following links:  
  
- [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
- [BrushNet and Stable Diffusion](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n)
- [AlphaCLIP](https://drive.google.com/file/d/1PvMJfg7nSVr98FfeBYSDAnbntGPWeDY6/view?usp=sharing)
- [DINO](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
