import copy
from typing import Optional, Union
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
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
            dino_vit = vit_large(**vit_kwargs)

            # Fetch weights
            dinov2_weights = torch.load(dinov2_weights, map_location="cpu")
            dinov2_weights = dinov2_weights["teacher"]
            dinov2_weights = {k.replace("module.", ""): v for k, v in dinov2_weights.items()}
            dinov2_weights = {k.replace("backbone.", ""): v for k, v in dinov2_weights.items()}
            
            # Also initialize v2, required later.
            from .transformer import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            dino_vit_v2 = vit_large(**vit_kwargs)
            dinov2_weights_v2 = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            dino_vit_v2.load_state_dict(dinov2_weights_v2, strict=False)
            dino_vit_v2.eval()
            if amp:
                dino_vit_v2 = dino_vit_v2.to(amp_dtype)
            self.dino_vit_v2 = [dino_vit_v2]
            
            
        elif self.dino_version == "v2":
            # Load model
            from .transformer import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            dino_vit = vit_large(**vit_kwargs)
            
            # Fetch weights
            if dinov2_weights is None:
                dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            elif type(dinov2_weights)==str and dinov2_weights.startswith("https://"):
                dinov2_weights = torch.hub.load_state_dict_from_url(dinov2_weights, map_location="cpu")

        # Load weights
        dino_vit.load_state_dict(dinov2_weights, strict=False)
        dino_vit.eval()
        
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        
        # Optionally cast to amp for v2 
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.dino_version == "v2" and self.amp:
            dino_vit = dino_vit.to(self.amp_dtype)

        self.dino_vit = [dino_vit] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def _preprocess(self, x):
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
        
        # Handle single image or batch of two images
        if x.shape[0] == 1:
            x = (x - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        elif x.shape[0] == 2:
            imA, imB = x
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
                self.dino_vit[0] = self.dino_vit[0].to(x.device)
                
                # Handle different dino versions. V1 requires removal of CLS token and explicit normalization.
                if self.dino_version == "v1":
                    # Forward features with v1
                    dino_features = self.dino_vit[0].get_intermediate_layers(x.to(self.amp_dtype), n=1)[0]
                    # dino_features = self.dino_vit[0].norm(dino_features)
                    features_16 = dino_features.permute(0, 2, 1)[:, :, 1:].reshape(B, 1024, H // 14, W // 14)
                    
                    # Also forward with v2
                    self.dino_vit_v2[0] = self.dino_vit_v2[0].to(x.device)
                    dino_features_v2 = self.dino_vit_v2[0].forward_features(x.to(self.amp_dtype))
                    features_16_v2 = dino_features_v2['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
                    
                    # Scale dino v1 features to match range of dino v2 features - super hacky if this works
                    # features_16_min, features_16_max = features_16.min(), features_16.max()
                    # features_16_v2_min, features_16_v2_max = features_16_v2.min(), features_16_v2.max()
                    
                    # features_16 = (features_16 - features_16_min) / (features_16_max - features_16_min) # Scale to 0-1
                    # features_16 = features_16 * (features_16_v2_max - features_16_v2_min) + features_16_v2_min # Scale to v2 range
                    
                elif self.dino_version == "v2":
                    dino_features = self.dino_vit[0].forward_features(x.to(self.amp_dtype))
                    features_16 = dino_features['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
                
                # if True:
                    # visualize_coarse_features(x, features_16, "/data/temporary/daan_s/rapid/results/cwz/test/coarse_features_from_pyramid_v1_scaled.png")
                
                del dino_features
                feature_pyramid[16] = features_16.to(self.amp_dtype)
                
        return feature_pyramid
    
def visualize_coarse_features(image, features, path):
    
    def get_plottable_features(features, mask):
        
        # Mask out background
        mask = cv2.resize(mask, (features.shape[0], features.shape[1])).astype(np.uint8)
        fg_features = copy.deepcopy(features)
        fg_features[mask==0] = 0
        
        # Fit PCA
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(fg_features.reshape(-1, fg_features.shape[-1]))
        
        # Normalize features to 0-1 range
        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
        for i in range(3):
            pca_features[:, i] = scaler.fit_transform(pca_features[:, i].reshape(-1, 1)).reshape(-1)
        
        # Convert to 2D
        pca_features = pca_features.reshape(features.shape[0], features.shape[1], -1)
        
        # Mask out any background artefacts
        bg = cv2.resize(mask, (pca_features.shape[0], pca_features.shape[1]))
        pca_features[bg == 0] = 0
        
        return pca_features
    
    # Get mask and image
    imA, imB = image
    imA = imA.permute(1, 2, 0).cpu().numpy()
    imB = imB.permute(1, 2, 0).cpu().numpy()
    maskA = (~np.all(imA == np.max(imA, axis=(0, 1)), axis=-1)).astype(np.uint8)
    maskB = (~np.all(imB == np.max(imB, axis=(0, 1)), axis=-1)).astype(np.uint8)
    
    featuresA, featuresB = features
    featuresA = featuresA.permute(1, 2, 0).cpu().numpy()
    featuresB = featuresB.permute(1, 2, 0).cpu().numpy()
    
    featuresA = get_plottable_features(featuresA, maskA)
    featuresB = get_plottable_features(featuresB, maskB)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(imA)
    plt.subplot(2,2,2)
    plt.imshow(featuresA)
    plt.subplot(2,2,3)
    plt.imshow(imB)
    plt.subplot(2,2,4)
    plt.imshow(featuresB)
    plt.savefig(path)
    plt.close()
    
    return
    