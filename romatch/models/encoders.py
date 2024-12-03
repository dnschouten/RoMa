from typing import Optional, Union
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
import sys
from romatch.utils.utils import get_autocast_params


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, dinov2_weights = None, amp_dtype = torch.float16, dino_version = "v2"):
        super().__init__()
        self.dino_version = dino_version
        
        # Add support for dino v1
        if self.dino_version == "v1":
            assert dinov2_weights is not None, "Path to dino v1 weights is required for dino v1"
            
            # Load model
            sys.path.append("/data/temporary/daan_s/rapid-dino/code/cpg-dino")
            from dino.models.vision_transformer import vit_large
            vit_kwargs = dict(
                img_size= 518,
                patch_size=14,
                drop_path_rate=0.1
            )
            dinov2_vitl14 = vit_large(**vit_kwargs)

            # Fetch weights
            dinov2_weights = torch.load(dinov2_weights, map_location="cpu")
            dinov2_weights = dinov2_weights["student"]
            dinov2_weights = {k.replace("module.", ""): v for k, v in dinov2_weights.items()}
            dinov2_weights = {k.replace("backbone.", ""): v for k, v in dinov2_weights.items()}
            
        elif self.dino_version == "v2":
            # Load model
            from .transformer import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            dinov2_vitl14 = vit_large(**vit_kwargs)
            
            # Fetch weights
            if dinov2_weights is None:
                dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            elif type(dinov2_weights)==str and dinov2_weights.startswith("https://"):
                dinov2_weights = torch.hub.load_state_dict_from_url(dinov2_weights, map_location="cpu")

        # Load weights
        dinov2_vitl14.load_state_dict(dinov2_weights, strict=False)
        dinov2_vitl14.eval()
        
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        
        # Optionally cast to amp for v2 
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.dino_version == "v2" and self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)

        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def _preprocess(self, x):
        # Normalize to imagenet mean
        imA, imB = x[0], x[1]
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
        imA = (imA - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        imB = (imB - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        x = torch.stack([imA, imB]).float()
        
        return x
    
    def forward(self, x, upsample = False):
        B, C, H, W = x.shape
        
        # Normalize to imagenet mean
        x = self._preprocess(x)
        
        # Extract fine-scaled features
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            with torch.no_grad():
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
                
                # Handle different dino versions. V1 requires removal of CLS token.
                if self.dino_version == "v1":
                    dinov2_features_16 = self.dinov2_vitl14[0].get_intermediate_layers(x.to(self.amp_dtype), n=1)[0]
                    features_16 = dinov2_features_16.permute(0, 2, 1)[:, :, 1:].reshape(B, 1024, H // 14, W // 14)
                elif self.dino_version == "v2":
                    dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                    features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
                
                del dinov2_features_16
                feature_pyramid[16] = features_16.to(self.amp_dtype)
                
        return feature_pyramid