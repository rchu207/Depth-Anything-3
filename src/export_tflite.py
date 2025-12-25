import argparse
import cv2
import glob
import numpy as np
import os
import torch
import qai_hub as hub
import depth_anything_3.cfg as da3cfg

from depth_anything_3.api import DepthAnything3

class DA3Wrapper(torch.nn.Module):
    """
    Thin wrapper around DepthAnything3 that accepts a single image tensor
    of shape (1, 3, H, W) and outputs depth (1, 1, H, W).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model  # underlying nn.Module

    def forward(self, image):
        # image: (1, 3, H, W) -> (B=1, N=1, 3, H, W)
        x = image.unsqueeze(0)
        out = self.model(x, None, None, export_feat_layers=[], infer_gs=False)
        return out["depth"]  # (1, H, W)


def main():
    # Parce input arguments
    parser = argparse.ArgumentParser(description='Depth Anything V3 ')
    parser.add_argument('--input-size', type=int, default=504)
    parser.add_argument('--load-from', type=str, default='depth-anything/da3-large')
    args = parser.parse_args()
    
    # Load model from Hugging Face Hub
    depth_anything = DepthAnything3.from_pretrained(args.load_from)
    depth_anything = depth_anything.to('cpu').eval()
    depth_anything = DA3Wrapper(depth_anything)
    model_full_name = args.load_from.replace('/', '-')
    print(f"Model {model_full_name} loaded on cpu")

    # Step 1: Trace model
    da3cfg.export_tflite_aihub = True
    input_shape = (1, 3, args.input_size, args.input_size)
    example_input = torch.rand(input_shape)
    traced_torch_model = torch.jit.trace(depth_anything, example_input)

    # Step 2: Compile model
    compile_job = hub.submit_compile_job(
        model=traced_torch_model,
        name=model_full_name,
        device=hub.Device("Samsung Galaxy S24 (Family)"),
        input_specs=dict(image=input_shape),
        options="--target_runtime tflite --force_channel_last_input image --output_names output --force_channel_last_output output",
    )
    assert isinstance(compile_job, hub.CompileJob)
    target_model = compile_job.get_target_model()
    assert isinstance(target_model, hub.Model)
    target_model.download(f'{model_full_name}.tflite')

    # Step 3: Profile on cloud-hosted device
    profile_job = hub.submit_profile_job(
        model=target_model,
        name=model_full_name,
        device=hub.Device("Samsung Galaxy S24 (Family)"),
    )
    assert isinstance(profile_job, hub.ProfileJob)
    # profile_job.wait()

    print(f"Model exported to {model_full_name}.tflite")


if __name__ == "__main__":
    main()
