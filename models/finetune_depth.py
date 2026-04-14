import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import utils3d
from moge.model.v1 import recover_focal_shift


def compute_scale_and_shift_full(prediction, target, mask):
    """Solve for optimal (scale, shift) to align prediction to target via least squares.

    Minimizes: sum(mask * (scale * prediction + shift - target)^2)

    Args:
        prediction (torch.Tensor | np.ndarray): Predicted depth map. Shape: (H, W).
        target (torch.Tensor | np.ndarray): Target depth map. Shape: (H, W).
        mask (torch.Tensor | np.ndarray): Float mask, 1.0 where valid. Shape: (H, W).

    Returns:
        scale (float): Optimal scale factor.
        shift (float): Optimal shift factor.
    """
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.cpu().numpy()
    if not isinstance(target, np.ndarray):
        target = target.cpu().numpy()
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


def compute_shift_only(prediction, target, mask, scale=1.0):
    """Solve for optimal shift with a fixed scale.

    Minimizes: sum(mask * (scale * prediction + shift - target)^2)

    Args:
        prediction (torch.Tensor | np.ndarray): Predicted depth map. Shape: (H, W).
        target (torch.Tensor | np.ndarray): Target depth map. Shape: (H, W).
        mask (torch.Tensor | np.ndarray): Float mask, 1.0 where valid. Shape: (H, W).
        scale (float): Fixed scale factor. Default: 1.0.

    Returns:
        shift (float): Optimal shift value.
    """
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.cpu().numpy()
    if not isinstance(target, np.ndarray):
        target = target.cpu().numpy()
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()

    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    sum_mask = np.sum(mask)
    if sum_mask == 0:
        return 0.0

    sum_target = np.sum(mask * target)
    sum_prediction = np.sum(mask * prediction)

    shift = (sum_target - scale * sum_prediction) / sum_mask
    return shift


def finetune_depth_model_step(model, target_depth, inpainted_image, a, b, mask_align=None, mask_cutoff=None, cutoff_depth=None):
    """Single optimization step for depth model finetuning.

    Runs MoGe inference on the input image, applies scale/shift to align with
    the target depth, and computes alignment + hinge loss.

    Args:
        model: MoGe depth model (moge.model.v1.MoGeModel). Must be in train mode
            with requires_grad=True for backprop.
        target_depth (torch.Tensor): Target depth map from Gaussian rendering (zbuf).
            Shape: (H, W). Values in metric depth, inf regions clipped to 1e9.
        inpainted_image (torch.Tensor): RGB image, pixel range [0, 1].
            Shape: (3, H, W). Typically H=W=512 or original resolution.
        a (float): Scale factor from compute_scale_and_shift_full().
            Aligns MoGe's relative depth to target's metric depth: aligned = a * moge_depth + b.
        b (float): Shift factor from compute_scale_and_shift_full().
        mask_align (torch.Tensor | None): Boolean mask for alignment loss region.
            Shape: (H, W). True where L1 loss should be computed.
            Typically: (zbuf < 9) & (zbuf > 0) & (~inf_mask).
        mask_cutoff (torch.Tensor | None): Boolean mask for hinge loss region.
            Shape: (H, W). True where depth should be pushed beyond cutoff_depth.
        cutoff_depth (float | torch.Tensor | None): Depth threshold for hinge loss.
            Penalizes predicted depth < cutoff_depth in mask_cutoff region.

    Returns:
        next_depth (torch.Tensor): Predicted depth after scale/shift. Shape: (H, W).
        total_loss (torch.Tensor): Scalar loss = L1_align + hinge_loss.
    """

    def infer(
        self,
        image: torch.Tensor,
        fov_x=None,
        resolution_level: int = 9,
        num_tokens: int = None,
        apply_mask: bool = True,
        force_projection: bool = True,
        use_fp16: bool = True,
    ):
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=use_fp16):
            output = self.forward(image, num_tokens)
        points, mask = output['points'], output['mask']

        mask_binary = mask > self.mask_threshold

        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        depth = points[..., 2] + shift[..., None, None]

        if force_projection:
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
        else:
            points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

        if omit_batch_dim:
            points = points.squeeze(0)
            intrinsics = intrinsics.squeeze(0)
            depth = depth.squeeze(0)
            mask_binary = mask_binary.squeeze(0)
            mask = mask.squeeze(0)

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'mask_prob': torch.sigmoid(mask)
        }

        return return_dict

    device = model.device
    next_depth = infer(model, inpainted_image.detach().cuda())['depth']
    inf_mask = (next_depth == torch.inf)
    next_depth[inf_mask] = 1e7
    next_depth = a * next_depth + b
    assert mask_align is not None
    mask_align = mask_align.cuda()
    # L1 loss for the mask_align region
    loss_align = F.l1_loss(target_depth.detach().cuda(), next_depth, reduction="none")
    if mask_align is not None and torch.any(mask_align):
        mask_align = mask_align.detach()
        loss_align = (loss_align * mask_align * (~inf_mask))[(mask_align * (~inf_mask)) > 0].mean()
    else:
        loss_align = torch.zeros(1).to(device)

    # Hinge loss for the mask_cutoff region
    if mask_cutoff is not None and cutoff_depth is not None and torch.any(mask_cutoff):
        hinge_loss = (cutoff_depth - next_depth).clamp(min=0)
        hinge_loss = F.l1_loss(hinge_loss, torch.zeros_like(hinge_loss), reduction="none")
        mask_cutoff = mask_cutoff.detach()
        hinge_loss = (hinge_loss * mask_cutoff)[mask_cutoff > 0].mean()
    else:
        hinge_loss = torch.zeros(1).to(device)

    total_loss = loss_align + hinge_loss
    if torch.isnan(total_loss):
        raise ValueError("Depth FT loss is NaN")

    return next_depth, total_loss


def finetune_depth_model(
    model,
    target_depth,
    input_image,
    a, b,
    mask_align=None,
    mask_cutoff=None,
    cutoff_depth=None,
    learning_rate=1e-6,
    num_steps=100,
    device="cuda",
):
    """Finetune MoGe depth model to align its predictions with a target depth map.

    Optimizes the model weights for `num_steps` via Adam, minimizing
    L1 alignment loss + hinge loss. Returns the best depth prediction
    (lowest loss across all steps).

    Args:
        model: MoGe depth model (moge.model.v1.MoGeModel).
            Will be set to train mode internally, restored to eval after.
        target_depth (torch.Tensor): Target depth map (e.g. zbuf from Gaussian splatting).
            Shape: (H, W). Values in metric depth.
        input_image (torch.Tensor): RGB image, pixel range [0, 1].
            Shape: (3, H, W).
        a (float): Scale factor. Computed via:
            a, b = compute_scale_and_shift_full(moge_depth, target_depth, mask)
        b (float): Shift factor.
        mask_align (torch.Tensor | None): Boolean mask for where to compute alignment loss.
            Shape: (H, W). If None, defaults to (target_depth > 0).
        mask_cutoff (torch.Tensor | None): Boolean mask for hinge loss region.
            Shape: (H, W).
        cutoff_depth (float | torch.Tensor | None): Depth threshold for hinge loss.
        learning_rate (float): Adam learning rate. Default: 1e-6.
        num_steps (int): Number of optimization steps. Default: 100.
        device (str): Device string. Default: "cuda".

    Returns:
        best_depth (torch.Tensor): Depth map with lowest loss. Shape: (H, W), on CUDA.

    Example:
        ```python
        from models.finetune_depth import (
            finetune_depth_model,
            compute_scale_and_shift_full,
        )

        # moge_depth: (H, W) from moge.model.infer(img)['depth']
        # zbuf: (H, W), the mask to align
        # img: (3, H, W) RGB [0, 1]
        # inf_mask: (H, W) bool, True where depth == inf

        mask = ((zbuf < big_threshold) & (zbuf > small_threshold) & (~inf_mask)).float() # the mask to align
        a, b = compute_scale_and_shift_full(moge_depth, zbuf, mask) # the scale and shift to align

        with torch.enable_grad():
            best_depth = finetune_depth_model(
                model=moge_model,
                target_depth=zbuf,
                input_image=img,
                a=a, b=b,
                mask_align=mask.bool(),
                learning_rate=1e-6,
                num_steps=100,
            )
        ```
    """
    target_depth.requires_grad_(True)
    input_image.requires_grad_(True)

    params = [{"params": model.parameters(), "lr": learning_rate}]
    optimizer = torch.optim.Adam(params)
    model.train()
    model.requires_grad_(True)
    if mask_align is None:
        mask_align = target_depth > 0

    progress_bar = tqdm(range(num_steps), leave=False)
    min_loss = torch.inf
    model.to(device)
    for _ in progress_bar:
        optimizer.zero_grad()

        next_depth, loss = finetune_depth_model_step(
            model,
            target_depth,
            input_image,
            a, b,
            mask_align=mask_align,
            mask_cutoff=mask_cutoff,
            cutoff_depth=cutoff_depth,
        )
        progress_bar.set_postfix(loss=loss.item(), a=a, b=b)
        loss.backward()
        optimizer.step()
        if loss < min_loss:
            min_loss = loss
            best_depth = next_depth

    model.eval()
    model.requires_grad_(False)
    return best_depth
