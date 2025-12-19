import glob, os
import torch
import cv2
import numpy as np
import OpenEXR as exr
import Imath
import argparse
import matplotlib
from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction


def export_to_mini_npz(
    depth: np.ndarray,  # [H,W]
    prediction: Prediction,
    output_dir: str,
    output_name: str,
):
    save_dict = {
        "depth": depth,
    }
    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(output_name))[0] + '.npz')
    np.savez_compressed(output_path, **save_dict)


def export_to_exr(
    depth: np.ndarray,  # [H,W]
    output_dir: str,
    output_name: str,
):
    height, width = depth.shape
    header = exr.Header(width, height)
    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    header['dataWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))
    header['displayWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))

    depth_buffer = depth.astype(np.float32).tobytes()

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(output_name))[0] + '.exr')
    exr_file = exr.OutputFile(output_path, header)
    exr_file.writePixels({'Z': depth_buffer})
    exr_file.close()


def export_to_depth_vis(
    image: np.ndarray,  # [H,W,C]
    depth: np.ndarray,  # [H,W]
    output_dir: str,
    output_name: str,
):
    percentile = 2
    cmap = "Spectral"

    image_vis = image.astype(np.uint8)

    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if valid_mask.sum() <= 10:
        depth_min = 0
    else:
        depth_min = np.percentile(depth[valid_mask], percentile)
    if valid_mask.sum() <= 10:
        depth_max = 0
    else:
        depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)

    depth_vis = img_colored_np
    depth_vis = depth_vis.astype(np.uint8)

    vis_image = np.concatenate([image_vis, depth_vis], axis=1)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(output_name))[0] + '.png')
    cv2.imwrite(output_path, vis_image)


if __name__ == '__main__':
    # Parce input arguments
    parser = argparse.ArgumentParser(description='Depth Anything V3 ')
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=504)
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--load-from', type=str, default='depth-anything/da3-large')
    args = parser.parse_args()

    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.load_from)
    model = model.to(device=device).eval()
    print(f"Model {args.load_from} loaded on {device}")

    if os.path.isfile(args.img_path):
        filenames = [args.img_path]
    else:
        filenames = sorted(glob.glob(os.path.join(args.img_path, '**/*'), recursive=True))
    os.makedirs(args.outdir, exist_ok=True)

    # Run inference on images
    for k, filename in enumerate(filenames):
        # print(f'Progress {k+1}/{len(filenames)}: {filename}')
        images = [filename, ]  # List of image paths, PIL Images, or numpy arrays
        # output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0])
        # prediction = model.inference(
        #     images,
        #     export_dir=output_path,
        #     export_format="mini_npz-depth_vis"
        # )
        prediction = model.inference(
            images,
        )

        # Access results
        print(prediction.processed_images.shape) # Processed images : [N, H, W, 3] uint8   array
        print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
        if prediction.conf is not None:
            print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
        if prediction.extrinsics is not None:
            print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
        if prediction.intrinsics is not None:
            print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
            print(prediction.intrinsics)

        # inferred depth map [H,W].
        infer_depth = prediction.depth.squeeze(0)
        print(filename, infer_depth.shape[0], infer_depth.shape[1], infer_depth.min(), infer_depth.max())

        # store inferred depth map to npz
        export_to_mini_npz(infer_depth, prediction, args.outdir, filename)

        # store inferred depth map to exr
        export_to_exr(infer_depth, args.outdir, filename)

        # store inferred depth map to colormap
        scaled_image = prediction.processed_images.squeeze(0)
        export_to_depth_vis(scaled_image, infer_depth, args.outdir, filename)
