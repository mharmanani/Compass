import torch
from torch import nn 


class TimmCNNWrapperForFeatures(torch.nn.Module):
    def __init__(self, cnn_model, pool=True):
        super().__init__()
        self.cnn_model = cnn_model
        self.pool = pool

    def forward(self, image):
        feature_map = self.cnn_model.forward_features(image)
        if self.pool:
            feature_map = torch.nn.functional.adaptive_avg_pool2d(feature_map, 1)[..., 0, 0]
        return feature_map


class VitWrapperForFeatureMaps(torch.nn.Module): 
    def __init__(self, vit): 
        super().__init__()
        self.vit = vit

    def forward(self, image): 
        B, C, H, W = image.shape
        vit_feature_map = self.vit(image, return_all_tokens=True)[:, 1:, :]
        B, npatches, C = vit_feature_map.shape
        vit_feature_map = vit_feature_map.reshape(
            B, int(npatches**0.5), int(npatches**0.5), C
        )
        vit_feature_map = vit_feature_map.permute(0, 3, 1, 2)  # B, C, H, W

        return vit_feature_map


class ImageModelWrapperForVideo(nn.Module): 
    def __init__(self, image_model): 
        super().__init__()
        self.image_model = image_model

    def forward(self, x): 
        assert x.ndim == 5 
        B, C, N, H, W = x.shape 
        x = x.permute(0, 2, 1, 3, 4) # B N C H W 
        x = x.reshape(B * N, C, H, W)
        x = self.image_model(x) # B * N, C, ... 
        new_shape = x.shape 
        unfolded_shape = B, N, *new_shape[1:]
        x = x.view(unfolded_shape)
        x = x.permute(0, 2, 1, 3, 4) # B C N ...
        return x


class FrozenModuleWrapper(nn.Module): 
    def __init__(self, module, frozen=True, allow_grad_passthrough=False, always_eval=True): 
        super().__init__()
        self.module = module
        self.frozen = frozen
        self.allow_grad_passthrough = allow_grad_passthrough
        self.always_eval = always_eval

        if self.frozen:
            for param in self.module.parameters():
                param.requires_grad = False

    def forward(self, *args, **kwargs): 
        grad_enabled = self.allow_grad_passthrough or not self.frozen
        with torch.set_grad_enabled(grad_enabled):
            if self.always_eval:
                self.module.eval()
            return self.module(*args, **kwargs)
