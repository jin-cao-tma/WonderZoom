"""Export a saved WonderZoom scene to a .splat file for web viewers (e.g. Spark).

The scenes written by the generation server ('save', X key) live in
runs/<example_name>/<YYYYmmdd-HHMMSS>/scenes/<example_name>_<NNN>.pth next to the session's config.yaml.

    # plain export: the scene as it is (no models needed)
    python tools/export_splat.py --scene runs/street/20250101-120000/scenes/street_000.pth --out street.splat

    # fill: re-generate the background hidden behind the inserted objects (loads the main models)
    python tools/export_splat.py --scene ... --out street.splat --mode fill

    # sky: fill (unless --no-fill) plus a regenerated sky layer, shifted by --shift X Y Z
    python tools/export_splat.py --scene ... --out ladybug.splat --mode sky --shift 5.1574e-3 -4.0808e-3 6.8847

'fill' and 'sky' are the paper-era yield_spark_splat_data / yield_spark_splat_data_sky helpers of the
generation server, ported unchanged apart from taking the scene and models as arguments.
"""
import copy
import os
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

WZ_ROOT = Path(__file__).resolve().parent.parent

# Known shifts used for the paper's web demos (centre of the object at the origin).
#   ladybug [5.1574e-3, -4.0808e-3, 6.8847]   bird [0.0306, 0.0250, 6.3509]
#   butterfly [0.0152, -0.0184, 9.9907]       lego [-0.0945, 0.0384, 9.9030]
#   fish [0, 0, 10.6012]                      lizard [-0.0220, 0.0285, 7.4713]


def _run_module():
    """The generation server module (train_gaussian, compute_3D_filter, render background, ...)."""
    if str(WZ_ROOT) not in sys.path:
        sys.path.insert(0, str(WZ_ROOT))
    import run as R
    return R


def export_splat_plain(gaussians, path):
    """Write the scene as it is."""
    gaussians.merge_all_to_trainable()
    gaussians.yield_splat_data(path)
    return gaussians


def export_splat_with_fill(gaussians, kf_gen, config, path):
    """Paper-era yield_spark_splat_data: re-generate the background behind non-main points from the
    origin view, merge it into the scene and write the result. Returns the exported model."""
    R = _run_module()
    import matplotlib.pyplot as plt
    from arguments_in import GSParams
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from util.utils import convert_pt3d_cam_to_3dgs_cam

    R.config = config
    xyz_scale = R.xyz_scale
    background = R.background
    opt = GSParams()

    gaus_copy = copy.deepcopy(gaussians)
    gaus_copy2 = copy.deepcopy(gaussians)
    gaus_copy.merge_all_to_trainable()
    mask = gaus_copy.point_labels == int(1e5)
    gaus_copy.delete_all_points(mask)
    gaus_copy.delete_all_points(gaus_copy.point_labels != 0)
    gaus_copy2.delete_all_points(mask)

    current_camera = kf_gen.get_camera_at_origin()
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
    render_result = render(tdgs_cam, gaus_copy, opt, background, filter_scale=False, config=config)
    depth = render_result["median_depth"][0] / xyz_scale
    image = render_result["render"]
    plt.imsave("./cache/image.png", image.detach().cpu().permute(1, 2, 0).numpy(), cmap="gray")
    render_result_no_filter = render(tdgs_cam, gaus_copy, opt, bg_color=background, filter_scale=True, config=config)
    mask_no_filter = (render_result_no_filter["final_opacity"] < 0.5).squeeze().detach().cpu()
    points_3d, colors, _, normals, imgs, cameras, focal_length, is_sky, all_depths, depth_align_masks, now_scale = \
        kf_gen.process_single_img_target(['./cache/image.png'], depth, mask_no_filter)

    traindata = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs, cameras, xyz_scale=xyz_scale,
                                                 use_no_loss_mask=False)

    gaus_new = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9, config=config)
    scene = Scene(traindata, gaus_new, opt, focal_length, is_sky, now_scale)
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
    gaus_copy.set_visible_and_restore_from_prev(tdgs_cam, opt)
    opt = GSParams()
    opt.iterations = 200
    opt.densify_from_iter = 1000
    opt.densify_until_iter = 1000
    trainCameras = scene.getTrainCameras().copy()

    R.compute_3D_filter(gaus_new, cameras=trainCameras, initialize_scaling=True)
    R.train_gaussian(gaus_new, scene, opt, xyz_scale=xyz_scale, newly_added_points=points_3d.shape[0],
                     no_loss_masks=[~mask_no_filter.squeeze().bool()])
    gaus_copy2.merge_gaussian(gaus_new)
    gaus_copy2.merge_all_to_trainable()
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
    gaus_copy.set_inscreen_points_to_visible(tdgs_cam)
    gaus_copy.merge_all_to_trainable()
    gaussians = gaus_copy2

    gaussians.yield_splat_data(path)
    return gaussians


def export_splat_sky(gaussians, kf_gen, config, path, shift, sky=True, wa=True):
    """Paper-era yield_spark_splat_data_sky: optional background fill (wa), optional regenerated sky
    layer (sky), then shift the scene by -shift and write it. Returns the exported model."""
    R = _run_module()
    import matplotlib.pyplot as plt
    from arguments_in import GSParams
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from util.utils import convert_pt3d_cam_to_3dgs_cam

    R.config = config
    xyz_scale = R.xyz_scale
    background = R.background
    opt = GSParams()

    gaus_copy = copy.deepcopy(gaussians)
    gaus_copy2 = copy.deepcopy(gaussians)
    if wa:
        gaus_copy.merge_all_to_trainable()
        mask = gaus_copy.point_labels == int(1e5)
        gaus_copy.delete_all_points(mask)
        gaus_copy.delete_all_points(gaus_copy.point_labels != 0)
        gaus_copy2.delete_all_points(mask)

        current_camera = kf_gen.get_camera_at_origin()
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
        render_result = render(tdgs_cam, gaussians, opt, background, filter_scale=False, config=config)
        depth = render_result["median_depth"][0] / xyz_scale
        image = render_result["render"]
        plt.imsave("./cache/image.png", image.detach().cpu().permute(1, 2, 0).numpy(), cmap="gray")
        render_result_no_filter = render(tdgs_cam, gaus_copy, opt, bg_color=background, filter_scale=True, config=config)
        mask_no_filter = (render_result_no_filter["final_opacity"] < 0.5).squeeze().detach().cpu()
        points_3d, colors, _, normals, imgs, cameras, focal_length, is_sky, all_depths, depth_align_masks, now_scale = \
            kf_gen.process_single_img_target(['./cache/image.png'], depth, mask_no_filter)

        traindata = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs, cameras, xyz_scale=xyz_scale,
                                                     use_no_loss_mask=False)

        gaus_new = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9, config=config)
        scene = Scene(traindata, gaus_new, opt, focal_length, is_sky, now_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
        gaus_copy.set_visible_and_restore_from_prev(tdgs_cam, opt)
        opt = GSParams()
        opt.iterations = 200
        opt.densify_from_iter = 1000
        opt.densify_until_iter = 1000
        trainCameras = scene.getTrainCameras().copy()

        R.compute_3D_filter(gaus_new, cameras=trainCameras, initialize_scaling=True)
        R.train_gaussian(gaus_new, scene, opt, xyz_scale=xyz_scale, newly_added_points=points_3d.shape[0],
                         no_loss_masks=[~mask_no_filter.squeeze().bool()])
        gaus_copy2.merge_gaussian(gaus_new)
        gaus_copy2.merge_all_to_trainable()
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
        gaus_copy.set_inscreen_points_to_visible(tdgs_cam)
        gaus_copy.merge_all_to_trainable()
        gaussians = gaus_copy2

    if sky:
        if not wa:
            # The sky layer is built from the origin view; the fill pass above normally renders it.
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
            image = render(tdgs_cam, gaussians, opt, background, filter_scale=False, config=config)["render"]
            plt.imsave("./cache/image.png", image.detach().cpu().permute(1, 2, 0).numpy(), cmap="gray")
        sky_mask = gaussians._xyz[:, 2] > 999
        gaussians.delete_all_points(sky_mask)

        points_3d, colors, _, normals, imgs, cameras, focal_length, is_sky, all_depths, depth_align_masks, now_scale = \
            kf_gen.process_single_img_sky(['./cache/image.png'])

        traindata = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs, cameras, xyz_scale=xyz_scale,
                                                     use_no_loss_mask=False)

        gaus_new = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9, config=config)
        scene = Scene(traindata, gaus_new, opt, focal_length, is_sky, now_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
        gaus_copy.set_visible_and_restore_from_prev(tdgs_cam, opt)
        opt = GSParams()
        opt.iterations = 200
        opt.densify_from_iter = 1000
        opt.densify_until_iter = 1000
        trainCameras = scene.getTrainCameras().copy()

        R.compute_3D_filter(gaus_new, cameras=trainCameras, initialize_scaling=True)
        R.train_gaussian(gaus_new, scene, opt, xyz_scale=xyz_scale, newly_added_points=points_3d.shape[0])
        gaussians.merge_gaussian(gaus_new)
        gaussians.merge_all_to_trainable()
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
        gaus_copy.set_inscreen_points_to_visible(tdgs_cam)
        gaus_copy.merge_all_to_trainable()
        # The paper-era helper ended with 'gaussians = gaus_copy2': a no-op after the fill pass (wa),
        # but without it the sky layer was dropped, so it is not repeated here.

    xyz_temp = gaussians._xyz.clone().detach()
    xyz_temp[:, 0] = xyz_temp[:, 0] - shift[0]
    xyz_temp[:, 1] = xyz_temp[:, 1] - shift[1]
    xyz_temp[:, 2] = xyz_temp[:, 2] - shift[2]
    gaussians._xyz = xyz_temp

    gaussians.yield_splat_data(path)
    return gaussians


def load_main_models(config, services_config="config/services.yaml"):
    """Build the VideoGaussianProcessor with the same main models as run.py (needed by fill / sky)."""
    import torch
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
    from marigold_lcm.marigold_pipeline import MarigoldNormalsPipeline
    from models.vdm_model import VideoGaussianProcessor
    from services import load_services_config
    from util.segment_utils import create_mask_generator_repvit

    services_cfg = load_services_config(main_path=services_config, local_path='config/services.local.yaml',
                                        overrides=dict(no_services=True))
    mask_generator = create_mask_generator_repvit(services_cfg.main_models.repvit_sam_checkpoint)
    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to("cuda")
    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16).to(config["device"])
    return VideoGaussianProcessor(config=config, segment_model=segment_model, segment_processor=segment_processor,
                                  normal_estimator=normal_estimator, mask_generator=mask_generator, moge=None,
                                  grounded_sam=None)


def main():
    parser = ArgumentParser(description="Export a saved WonderZoom scene (.pth) to a .splat file")
    parser.add_argument("--scene", required=True, help="scene saved by the server (<session>/scenes/<name>_<NNN>.pth)")
    parser.add_argument("--config", default=None,
                        help="session config.yaml (default: <session>/config.yaml next to the scenes/ folder)")
    parser.add_argument("--out", required=True, help="output .splat path")
    parser.add_argument("--mode", default="plain", choices=["plain", "fill", "sky"],
                        help="plain: the scene as saved; fill: regenerate the background behind objects; "
                             "sky: fill plus a regenerated sky layer and a shift")
    parser.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                        help="sky mode: translation subtracted from every point")
    parser.add_argument("--no-fill", dest="fill", action="store_false", help="sky mode: skip the background fill")
    parser.add_argument("--no-sky", dest="sky", action="store_false", help="sky mode: skip the sky layer")
    parser.add_argument("--services_config", default="config/services.yaml")
    parser.add_argument("--work_dir", default=None, help="scratch directory for intermediate images (default: a temp dir)")
    args = parser.parse_args()

    # Resolve user paths before importing run (which changes into the repository root).
    scene_path = os.path.abspath(args.scene)
    out_path = os.path.abspath(args.out)
    config_path = os.path.abspath(args.config) if args.config else \
        os.path.join(os.path.dirname(os.path.dirname(scene_path)), "config.yaml")
    if not os.path.isfile(scene_path):
        sys.exit(f"scene not found: {scene_path}")
    if not os.path.isfile(config_path):
        sys.exit(f"config not found: {config_path} (pass --config)")
    services_config = args.services_config if os.path.isabs(args.services_config) else \
        str(WZ_ROOT / args.services_config)
    work_dir = os.path.abspath(args.work_dir) if args.work_dir else tempfile.mkdtemp(prefix="wz_export_")
    os.makedirs(os.path.join(work_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    R = _run_module()
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    R.config = config
    gaussians = R.load_gaussian_with_global_labels(scene_path, config)

    kf_gen = None
    if args.mode != "plain":
        kf_gen = load_main_models(config, services_config)
    os.chdir(work_dir)  # the helpers write ./cache/image.png
    if args.mode == "plain":
        export_splat_plain(gaussians, out_path)
    elif args.mode == "fill":
        export_splat_with_fill(gaussians, kf_gen, config, out_path)
    else:
        export_splat_sky(gaussians, kf_gen, config, out_path, args.shift, sky=args.sky, wa=args.fill)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
