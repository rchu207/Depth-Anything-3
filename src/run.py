import glob, os
import torch
import cv2
import numpy as np
import OpenEXR as exr
import Imath
import argparse
import matplotlib
from depth_anything_3.api import DepthAnything3

if __name__ == '__main__':
    # Parce input arguments
    parser = argparse.ArgumentParser(description='Depth Anything V3 ')
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--load-from', type=str, default='depth-anything/da3-large')
    args = parser.parse_args()

    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.load_from)
    model = model.to(device=device)

    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral')

    # Run inference on images
    if os.path.isfile(args.img_path):
        filenames = [args.img_path]
    else:
        filenames = sorted(glob.glob(os.path.join(args.img_path, '**/*'), recursive=True))
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        images = [filename, ]  # List of image paths, PIL Images, or numpy arrays
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

        # store inferred depth map to npy
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.npy')
        np.save(output_path, infer_depth)

        # store normalized depth map to exr
        exr_depth = infer_depth / infer_depth.max()
        exr_depth = exr_depth.astype(np.float32).tobytes()
        height, width = infer_depth.shape
        header = exr.Header(width, height)
        header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        header['dataWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))
        header['displayWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.exr')
        exr_file = exr.OutputFile(output_path, header)
        exr_file.writePixels({'Z': exr_depth})
        exr_file.close()

        # store normalized depth map to png
        heat_depth = (infer_depth - infer_depth.min()) / (infer_depth.max() - infer_depth.min()) * 255.0
        heat_depth = heat_depth.astype(np.uint8)
        heat_depth = (cmap(heat_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        cv2.imwrite(output_path, heat_depth)
