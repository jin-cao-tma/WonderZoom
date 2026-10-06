import torch
import torch.nn as nn

from .moge.model.v1 import *
from .moge.model.v2 import *
class MoGe(nn.Module):
    
    def __init__(self, cache_dir=None, revision='979e84da9415762c30e6c0cf8dc0962896c793df'):
        super().__init__()
        # cache_dir=None uses the standard Hugging Face cache; revision pins Ruicheng/moge-vitl (pins.env HF_REV_MOGE_VITL).
        self.model = MoGeModel.from_pretrained(
            'Ruicheng/moge-vitl', cache_dir=cache_dir, revision=revision).eval()
        # self.model_v2 = MoGeModelv2.from_pretrained(
        #     'Ruicheng/moge-2-vitl').eval()


    @torch.no_grad()
    def forward_image(self, image: torch.Tensor, **kwargs):
        # image: b, 3, h, w 0,1
        output = self.model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
        points = output['points'] # b,h,w,3
        masks = output['mask'] # b,h,w
        return points, masks