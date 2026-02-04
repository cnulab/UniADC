import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import FeatureFusionBlock_custom
import numpy as np
from dinov3.hub.backbones import load_dinov3_model
from dinov3.hub.dinotxt import DinoV3TextEncoder
from dinov3.eval.text.vision_tower import VisionHead


PROMPT_TEMPLATES = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "a cropped image featuring {}",
)

class ProjectReadout(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectReadout, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, out_features, bias=False), nn.ReLU())

    def forward(self, x):
        return self.project(x)


def _make_fusion_block(channel, use_bn):
    return FeatureFusionBlock_custom(
        channel,
        activation=nn.ReLU(False),
        bn=use_bn,
    )


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        vision_pretrained_weights,
        text_pretrained_weights,
        labels = None,
        device ='cuda',
        use_bn = True,
    ):
        super(Discriminator, self).__init__()

        self.target_layers = [2, 5, 8, 11]
        self.device = device

        self.vision_encoder = load_dinov3_model('dinov3_vitb16',
                                    layers_to_extract_from = self.target_layers,
                                    pretrained_weight_path = vision_pretrained_weights)

        self.vision_encoder.eval()
        self.vision_dim = 768

        self.text_encoder = DinoV3TextEncoder(device)
        self.text_encoder.eval()
        self.text_dim = 1024

        state_dict = torch.load(text_pretrained_weights)
        state_dict = {key: state_dict[key] for key in state_dict if key.startswith('text_model')}
        self.text_encoder.load_state_dict(state_dict, strict=False)

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.necks = nn.ParameterList([ProjectReadout(self.vision_dim, self.text_dim)
                                       for _ in range(len(self.target_layers))])

        self.fusionnets = nn.ParameterList([])
        self.fusionnets.append(_make_fusion_block(self.text_dim, use_bn))
        self.fusionnets.append(_make_fusion_block(self.text_dim, use_bn))
        self.fusionnets.append(_make_fusion_block(self.text_dim, use_bn))
        self.fusionnets.append(_make_fusion_block(self.text_dim, use_bn))

        self.epsilon = nn.Parameter(torch.tensor(1.0))

        if labels is not None:
            with torch.no_grad():
                self.labels_features = self.build_text_embeddings(labels, PROMPT_TEMPLATES)

    def preprocess_features(self, en_list):
        feats = []
        for idx, (patch_tokens, cls_token) in enumerate(en_list):
            feat = self.necks[idx](patch_tokens)
            B, L, C = feat.shape
            side = int(math.sqrt(L))
            feats.append(feat.permute(0, 2, 1).view((B, C, side, side)))
        return feats

    @torch.no_grad()
    def build_text_embeddings(self, class_names, prompt_templates ) -> torch.Tensor:
        prompts, owners = [], []
        for c_idx, name in enumerate(class_names):
            for tpl in prompt_templates:
                prompts.append(tpl.format(name))
                owners.append(c_idx)
        embs = self.text_encoder.encode_text(prompts).to(self.device)
        C, D = len(class_names), embs.size(1)
        agg = torch.zeros(C, D, device=embs.device)
        cnt = torch.zeros(C,    device=embs.device)
        for i, c in enumerate(owners):
            agg[c] += embs[i]
            cnt[c] += 1
        return F.normalize(agg / cnt.unsqueeze(1), p=2, dim=1)


    def forward(self, x, labelset=''):
        with torch.no_grad():
            if labelset != '':
                self.labels_features = self.build_text_embeddings(labelset, PROMPT_TEMPLATES)

        labels_features = self.labels_features.to(device=x.device, dtype=x.dtype)
        en_list = self.vision_encoder.get_intermediate_layers(x, n=self.target_layers, return_class_token = True, norm = True)
        layer_1, layer_2, layer_3, layer_4 = self.preprocess_features(en_list)

        path_4 = self.fusionnets[0](layer_4)
        path_3 = self.fusionnets[1](path_4, layer_3)
        path_2 = self.fusionnets[2](path_3, layer_2)
        image_features = self.fusionnets[3](path_2, layer_1)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.text_dim)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        labels_features = labels_features / labels_features.norm(dim=-1, keepdim=True)

        classification_maps = (image_features @ labels_features.t() / self.epsilon)
        classification_maps = classification_maps.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        classification_maps = F.interpolate(classification_maps, size=x.shape[2:], mode='bilinear')
        detection_maps = torch.mean(classification_maps, dim=1, keepdim=True)

        return detection_maps, classification_maps
