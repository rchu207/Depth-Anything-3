import argparse
import cv2
import glob
import numpy as np
import os
import torch
import qai_hub as hub
import depth_anything_3.cfg as da3cfg
from PIL import Image

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

    export_tflite = False
    export_quantized_onnx = False

    # Step 1: Trace model
    da3cfg.export_tflite_aihub = True
    input_shape = (1, 3, args.input_size, args.input_size)
    example_input = torch.rand(input_shape)
    traced_torch_model = torch.jit.trace(depth_anything, example_input)

    if export_tflite:
        # Step 2: Compile model
        # Use FP16 weights, --quantize_weight_type float16
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
        # re-used uploaded model
        # target_model = hub.get_model("mq251vk0q")
        profile_job = hub.submit_profile_job(
            model=target_model,
            name=model_full_name,
            device=hub.Device("Samsung Galaxy S24 (Family)"),
        )
        assert isinstance(profile_job, hub.ProfileJob)
        profile_job.wait()

        print(f"Model exported to {model_full_name}.tflite")

    if export_quantized_onnx:
        # Step 2: Compile model to ONNX
        if True:
            print(f"Compile model to ONNX")
            compile_onnx_job = hub.submit_compile_job(
                model=traced_torch_model,
                name=model_full_name,
                device=hub.Device("Samsung Galaxy S24 (Family)"),
                input_specs=dict(image=input_shape),
                options="--target_runtime onnx",
            )
            assert isinstance(compile_onnx_job, hub.CompileJob)
            unquantized_onnx_model = compile_onnx_job.get_target_model()
            assert isinstance(unquantized_onnx_model, hub.Model)

        # Step 3: Load and pre-process downloaded calibration data
        # This transform is required for PyTorch imagenet classifiers
        # Source: https://pytorch.org/hub/pytorch_vision_resnet/
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        sample_inputs = []

        images_dir = "imagenette_samples"
        for image_path in os.listdir(images_dir):
            image = Image.open(os.path.join(images_dir, image_path))
            image = image.convert("RGB").resize(input_shape[2:])
            sample_input = np.array(image).astype(np.float32) / 255.0
            sample_input = np.expand_dims(np.transpose(sample_input, (2, 0, 1)), 0)
            sample_inputs.append(((sample_input - mean) / std).astype(np.float32))
        calibration_data = dict(image=sample_inputs)

        # Step 4: Quantize the model
        if True:
            print(f"Quantize ONNX model")
            # re-used unquantized ONNX model
            # unquantized_onnx_model = hub.get_model("mm62wg8dn")
            quantize_job = hub.submit_quantize_job(
                model=unquantized_onnx_model,
                calibration_data=calibration_data,
                weights_dtype=hub.QuantizeDtype.INT8,
                activations_dtype=hub.QuantizeDtype.INT8,
            )
            quantized_onnx_model = quantize_job.get_target_model()
            assert isinstance(quantized_onnx_model, hub.Model)

        # Step 5. Compile to target runtime (TFLite)
        if True:
            print(f"Compile model to TFLite")
            # re-used quantized ONNX model
            # quantized_onnx_model = hub.get_model("mm62wg04n")
            # use int8 io, --quantize_io
            compile_tflite_job = hub.submit_compile_job(
                model=quantized_onnx_model,
                name=model_full_name,
                device=hub.Device("Samsung Galaxy S24 (Family)"),
                options="--target_runtime tflite --force_channel_last_input image --output_names output --force_channel_last_output output",
            )
            assert isinstance(compile_tflite_job, hub.CompileJob)
            quantized_tflite_model = compile_tflite_job.get_target_model()
            assert isinstance(quantized_tflite_model, hub.Model)
            quantized_tflite_model.download(f'{model_full_name}_quantized.tflite')

            print(f'{model_full_name}_quantized.tflite')

            profile_job = hub.submit_profile_job(
                model=quantized_tflite_model,
                name=model_full_name,
                device=hub.Device("Samsung Galaxy S24 (Family)"),
            )
            assert isinstance(profile_job, hub.ProfileJob)

if __name__ == "__main__":
    main()
