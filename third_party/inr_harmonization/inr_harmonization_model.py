# INR-Harmonization inference wrapper used by WonderZoom for optional object harmonization.
#
# Adapted from WindVChen/INR-Harmonization (Apache-2.0), inference_for_arbitrary_resolution_image.py.
# scripts/setup_third_party.sh copies this file into external/INR-Harmonization after applying
# third_party/inr_harmonization/inr_harmonization.patch, which moves utils/misc.py to a
# top-level misc.py (WonderZoom has its own 'utils' package).
#
# The HRNet ImageNet initialisation is not needed: Resolution_RAW_iHarmony4.pth contains the
# full network weights. The patched model/backbone.py only loads it when the environment
# variable INR_HRNET_PRETRAINED points to the file.
#
# Usage (from the WonderZoom main process, with external/INR-Harmonization on sys.path):
#     model = INRHarmonizationModel(pretrained_path, device='cuda')
#     model.inference(composite_path, mask_path, output_path)
import torch
import cv2
import numpy as np
import os
import time
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import albumentations
from albumentations import Resize
import math

from model.build_model import build_model
from misc import normalize, prepare_cooridinate_input, customRandomCrop, get_mgrid


class INRHarmonizationModel:
    def __init__(self, pretrained_path, device='cuda'):
        """Build the INR-Harmonization model and load the pretrained weights."""
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"INR-Harmonization checkpoint not found: {pretrained_path}. "
                "Download Resolution_RAW_iHarmony4.pth (see scripts/download_checkpoints.sh --objects).")
        self.device = device

        # Inference options of the 'Resolution_RAW_iHarmony4' model.
        class Opt:
            def __init__(self):
                self.split_num = 2
                self.base_size = 256
                self.input_size = 256
                self.INR_input_size = 256
                self.INR_MLP_dim = 32
                self.LUT_dim = 7
                self.activation = 'leakyrelu_pe'
                self.param_factorize_dim = 10
                self.embedding_type = "CIPS_embed"
                self.INRDecode = True
                self.isMoreINRInput = True
                self.hr_train = True
                self.isFullRes = True
                self.transform_mean = [.5, .5, .5]
                self.transform_var = [.5, .5, .5]
                self.device = device
                self.batch_size = 1
                self.workers = 0

        self.opt = Opt()

        print("Loading INR-Harmonization model...")
        self.model = build_model(self.opt).to(device)
        # The release checkpoint also stores training state, so it is not a weights-only file.
        load_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)['model']
        self.model.load_state_dict(load_dict, strict=False)
        self.model.eval()
        print("INR-Harmonization model loaded.")

    def to(self, device):
        """Move the model to another device (used to park it in host RAM). Returns self."""
        self.model.to(device)
        self.device = device
        self.opt.device = device
        return self

    @torch.no_grad()
    def inference(self, composite_image_path, mask_path, output_path=None):
        """Harmonize a composite image given its foreground mask; optionally save the result."""
        composite_image = cv2.imread(composite_image_path)
        if composite_image is None:
            raise ValueError(f"Cannot load composite image from {composite_image_path}")
        composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        if mask is None:
            raise ValueError(f"Cannot load mask from {mask_path}")
        mask = np.squeeze(mask[:, :, 0]).astype(np.float32) / 255.

        # Same procedure as upstream inference_for_arbitrary_resolution_image.py.
        result = inference(self.model, self.opt, composite_image, mask)

        if output_path:
            cv2.imwrite(output_path, result)

        return result


@torch.no_grad()
def inference(model, opt, composite_image=None, mask=None):
    model.eval()

    "dataset here is actually consisted of several patches of a single image."
    singledataset = single_image_dataset(opt, composite_image, mask)

    single_data_loader = DataLoader(singledataset, opt.batch_size, shuffle=False, drop_last=False, pin_memory=True,
                                    num_workers=opt.workers, persistent_workers=False if composite_image is not None else True)

    "Init a pure black image with the same size as the input image."
    init_img = np.zeros_like(singledataset.composite_image)

    time_all = 0

    for step, batch in tqdm.tqdm(enumerate(single_data_loader)):
        composite_image = [batch[f'composite_image{name}'].to(opt.device) for name in range(4)]
        mask = [batch[f'mask{name}'].to(opt.device) for name in range(4)]
        coordinate_map = [batch[f'coordinate_map{name}'].to(opt.device) for name in range(4)]
        start_points = batch['start_point']

        if opt.batch_size == 1:
            start_points = [torch.cat(start_points)]

        fg_INR_coordinates = coordinate_map[1:]

        try:
            if step == 0:  # This is for CUDA Kernel Warm-up, or the first inference step will be quite slow.
                fg_content_bg_appearance_construct, _, lut_transform_image = model(
                    composite_image,
                    mask,
                    fg_INR_coordinates,
                )
                print("Ready for harmonization...")

            # Timing only. Upstream also reset the CUDA peak-memory counters here, which would
            # clobber the peak-memory measurements of the WonderZoom process, so that is omitted.
            if opt.device == "cuda":
                start_time = time.time()
                torch.cuda.synchronize()
            fg_content_bg_appearance_construct, _, lut_transform_image = model(
                composite_image,
                mask,
                fg_INR_coordinates,
            )
            if opt.device == "cuda":
                torch.cuda.synchronize()
                end_time = time.time()
                time_all += (end_time - start_time)
            print(f'progress: {step} / {len(single_data_loader)}')
        except Exception as e:
            raise RuntimeError(
                f'The image resolution is large. Please increase the `split_num` value. Your current set is {opt.split_num}') from e

        "Assemble the every patch's harmonized result into the final whole image."
        for id in range(len(fg_INR_coordinates[0])):
            pred_fg_image = fg_content_bg_appearance_construct[-1][id]
            pred_harmonized_image = pred_fg_image * (mask[1][id] > 100 / 255.) + composite_image[1][id] * (
                ~(mask[1][id] > 100 / 255.))

            pred_harmonized_tmp = cv2.cvtColor(
                normalize(pred_harmonized_image.unsqueeze(0), opt, 'inv')[0].permute(1, 2, 0).cpu().mul_(255.).clamp_(
                    0., 255.).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

            init_img[start_points[id][0]:start_points[id][0] + singledataset.split_height_resolution,
            start_points[id][1]:start_points[id][1] + singledataset.split_width_resolution] = pred_harmonized_tmp

    if opt.device == "cuda":
        print(f'Inference time: {time_all}')
    return init_img


class single_image_dataset(torch.utils.data.Dataset):
    def __init__(self, opt, composite_image=None, mask=None):
        super().__init__()

        self.opt = opt

        if composite_image is None:
            composite_image = cv2.imread(opt.composite_image)
            composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        self.composite_image = composite_image

        if mask is None:
            mask = cv2.imread(opt.mask)
            mask = mask[:, :, 0].astype(np.float32) / 255.
        self.mask = mask

        self.torch_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        self.INR_dataset = Implicit2DGenerator(opt, 'Val')

        self.split_width_resolution = composite_image.shape[1] // opt.split_num
        self.split_height_resolution = composite_image.shape[0] // opt.split_num

        self.split_width_resolution = self.split_height_resolution = min(self.split_width_resolution,
                                                                         self.split_height_resolution)

        if self.split_width_resolution % 4 != 0:
            self.split_width_resolution = self.split_width_resolution + (4 - self.split_width_resolution % 4)

        if self.split_height_resolution % 4 != 0:
            self.split_height_resolution = self.split_height_resolution + (4 - self.split_height_resolution % 4)

        self.num_w = math.ceil(composite_image.shape[1] / self.split_width_resolution)
        self.num_h = math.ceil(composite_image.shape[0] / self.split_height_resolution)

        self.split_start_point = []

        "Split the image into several parts."
        for i in range(self.num_h):
            for j in range(self.num_w):
                if i == composite_image.shape[0] // self.split_height_resolution:
                    if j == composite_image.shape[1] // self.split_width_resolution:
                        self.split_start_point.append((composite_image.shape[0] - self.split_height_resolution,
                                                       composite_image.shape[1] - self.split_width_resolution))
                    else:
                        self.split_start_point.append(
                            (composite_image.shape[0] - self.split_height_resolution, j * self.split_width_resolution))
                else:
                    if j == composite_image.shape[1] // self.split_width_resolution:
                        self.split_start_point.append(
                            (i * self.split_height_resolution, composite_image.shape[1] - self.split_width_resolution))
                    else:
                        self.split_start_point.append(
                            (i * self.split_height_resolution, j * self.split_width_resolution))

        assert len(self.split_start_point) == self.num_w * self.num_h

        print(
            f"The image will be split into {self.num_h} pieces in height, and {self.num_w} pieces in width. Totally {self.num_h * self.num_w} patches.")
        print(f"The final resolution of each patch is {self.split_height_resolution} x {self.split_width_resolution}")

    def __len__(self):
        return self.num_w * self.num_h

    def __getitem__(self, idx):
        composite_image = self.composite_image

        mask = self.mask

        full_coord = prepare_cooridinate_input(mask).transpose(1, 2, 0)

        tmp_transform = albumentations.Compose([Resize(self.opt.base_size, self.opt.base_size)],
                                               additional_targets={'object_mask': 'image'})
        transform_out = tmp_transform(image=composite_image, object_mask=mask)
        compos_list = [self.torch_transforms(transform_out['image'])]
        mask_list = [
            torchvision.transforms.ToTensor()(transform_out['object_mask'][..., np.newaxis].astype(np.float32))]
        coord_map_list = []

        if composite_image.shape[0] != self.split_height_resolution:
            c_h = self.split_start_point[idx][0] / (composite_image.shape[0] - self.split_height_resolution)
        else:
            c_h = 0
        if composite_image.shape[1] != self.split_width_resolution:
            c_w = self.split_start_point[idx][1] / (composite_image.shape[1] - self.split_width_resolution)
        else:
            c_w = 0
        transform_out, c_h, c_w = customRandomCrop([composite_image, mask, full_coord],
                                                   self.split_height_resolution, self.split_width_resolution, c_h, c_w)

        compos_list.append(self.torch_transforms(transform_out[0]))
        mask_list.append(
            torchvision.transforms.ToTensor()(transform_out[1][..., np.newaxis].astype(np.float32)))
        coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[2]))
        coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[2]))
        for n in range(2):
            tmp_comp = cv2.resize(composite_image, (
                composite_image.shape[1] // 2 ** (n + 1), composite_image.shape[0] // 2 ** (n + 1)))
            tmp_mask = cv2.resize(mask, (mask.shape[1] // 2 ** (n + 1), mask.shape[0] // 2 ** (n + 1)))
            tmp_coord = prepare_cooridinate_input(tmp_mask).transpose(1, 2, 0)

            transform_out, c_h, c_w = customRandomCrop([tmp_comp, tmp_mask, tmp_coord],
                                                       self.split_height_resolution // 2 ** (n + 1),
                                                       self.split_width_resolution // 2 ** (n + 1), c_h, c_w)
            compos_list.append(self.torch_transforms(transform_out[0]))
            mask_list.append(
                torchvision.transforms.ToTensor()(transform_out[1][..., np.newaxis].astype(np.float32)))
            coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[2]))
        out_comp = compos_list
        out_mask = mask_list
        out_coord = coord_map_list

        fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB = self.INR_dataset.generator(
            self.torch_transforms, transform_out[0], transform_out[0], mask)

        return {
            'composite_image': out_comp,
            'mask': out_mask,
            'coordinate_map': out_coord,
            'composite_image0': out_comp[0],
            'mask0': out_mask[0],
            'coordinate_map0': out_coord[0],
            'composite_image1': out_comp[1],
            'mask1': out_mask[1],
            'coordinate_map1': out_coord[1],
            'composite_image2': out_comp[2],
            'mask2': out_mask[2],
            'coordinate_map2': out_coord[2],
            'composite_image3': out_comp[3],
            'mask3': out_mask[3],
            'coordinate_map3': out_coord[3],
            'fg_INR_coordinates': fg_INR_coordinates,
            'bg_INR_coordinates': bg_INR_coordinates,
            'fg_INR_RGB': fg_INR_RGB,
            'fg_transfer_INR_RGB': fg_transfer_INR_RGB,
            'bg_INR_RGB': bg_INR_RGB,
            'start_point': self.split_start_point[idx],
        }


class Implicit2DGenerator(object):
    def __init__(self, opt, mode):
        if mode == 'Train':
            sidelength = opt.INR_input_size
        elif mode == 'Val':
            sidelength = opt.input_size
        else:
            raise NotImplementedError

        self.mode = mode
        self.size = sidelength

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)

        self.mgrid = get_mgrid(sidelength)
        self.transform = albumentations.Resize(self.size, self.size)

    def generator(self, torch_transforms, composite_image, real_image, mask):
        composite_image = torch_transforms(self.transform(image=composite_image)['image'])
        real_image = torch_transforms(self.transform(image=real_image)['image'])

        fg_INR_RGB = composite_image.permute(1, 2, 0).contiguous().view(-1, 3)
        fg_transfer_INR_RGB = real_image.permute(1, 2, 0).contiguous().view(-1, 3)
        bg_INR_RGB = real_image.permute(1, 2, 0).contiguous().view(-1, 3)

        fg_INR_coordinates = self.mgrid
        bg_INR_coordinates = self.mgrid

        return fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB
