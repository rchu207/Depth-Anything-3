import glob, os
import torch
import cv2
import numpy as np
import OpenEXR as exr
import Imath
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3-large")
model = model.to(device=device)

# Run inference on images
images = ["image1.jpg", ]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(
    images,
    export_dir="output",
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)

# Access results
print(prediction.processed_images.shape) # Processed images : [N, H, W, 3] uint8   array
print(prediction.is_metric) 
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.depth.min()) 
print(prediction.depth.max()) 
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
print(prediction.intrinsics)

depth = (prediction.depth) / (prediction.depth.max()) * 255.0
depth = depth.astype(np.uint8)
depth = depth.squeeze()
depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
filename = images[0]
output_path = os.path.join("output", os.path.splitext(os.path.basename(filename))[0] + '_zero2max.png')
cv2.imwrite(output_path, depth)

depth = (prediction.depth - prediction.depth.min()) / (prediction.depth.max() - prediction.depth.min()) * 255.0
depth = depth.astype(np.uint8)
depth = depth.squeeze()
depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
filename = images[0]
output_path = os.path.join("output", os.path.splitext(os.path.basename(filename))[0] + '_min2max.png')
cv2.imwrite(output_path, depth)

focal = (prediction.intrinsics[0][0][0] + prediction.intrinsics[0][1][1]) / 2
print(focal)
metric_depth = focal * prediction.depth / 300
print(metric_depth.min()) 
print(metric_depth.max()) 

depth = (metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min()) * 255.0
depth = depth.astype(np.uint8)
depth = depth.squeeze()
depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
filename = images[0]
output_path = os.path.join("output", os.path.splitext(os.path.basename(filename))[0] + '_focal_min2max.png')
cv2.imwrite(output_path, depth)

# Define the header for a single-channel 'Z' depth map
exr_depth = prediction.depth.squeeze()
height, width = exr_depth.shape
header = exr.Header(width, height)
header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
header['dataWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))
header['displayWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))

# Create the output file object
output_path = os.path.join("output", os.path.splitext(os.path.basename(filename))[0] + '.exr')
exr_file = exr.OutputFile(output_path, header)

# Ensure the numpy array is in float32 format
exr_depth = exr_depth.astype(np.float32).tobytes()

# Write the pixel data to the 'Z' channel
exr_file.writePixels({'Z': exr_depth})

# Close the file
exr_file.close()
