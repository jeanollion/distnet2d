from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(path_root.joinpath("dataset_iterator").__str__())
sys.path.append(path_root.joinpath("distne2d").__str__())

import tensorflow as tf
import tifffile
import numpy as np

from distnet_2d.model import get_distnet_2d, architectures

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def convert_tensor_from_disk(filename):
    with open(filename, "rb") as f:
        tensor_bytes = f.read()
    tensor = tf.io.parse_tensor(tensor_bytes, out_type=tf.float32).numpy()
    tensor = np.transpose(tensor, [0, 3, 1, 2])
    print(f"shape: {tensor.shape} type: {type(tensor)}")
    tifffile.imwrite(filename.replace(".npy", ".tiff"), data=tensor)

def load_tensor_from_disk(filename):
    with open(filename, "rb") as f:
        tensor_bytes = f.read()
    return tf.io.parse_tensor(tensor_bytes, out_type=tf.float32).numpy()

i0 = load_tensor_from_disk("/data2/MotherMachineSpots/input0.npy")
i1 = load_tensor_from_disk("/data2/MotherMachineSpots/input1.npy")
i2 = load_tensor_from_disk("/data2/MotherMachineSpots/input2.npy")

dn = get_distnet_2d(
    arch=architectures.TemAD3(n_inputs=3, inference_gap_number=2, frame_window=9, spatial_dimensions=(384, 32), filters=192, temporal_attention=16, self_attention=16,
                              attention=16, attention_filters=64, attention_positional_encoding="EMBEDDING", skip_connections=False, early_downsampling=True, category_number=0, frame_max_distance=30,
                              predict_edm_derivatives=False, predict_cdm_derivatives=False))
dn.load_weights("/data2/MotherMachineSpots/spot_muth_tema_fw9d3f192_asa16x64_cdmwm.h5")
frames = np.arange(0, 19, dtype="float32")
frames = np.reshape(frames, (1, 1, 1, 19))
for b in range(i0.shape[0]):
    pred = dn.predict((i0[b:b+1], i1[b:b+1], i2[b:b+1], frames))
    #pred = dn.predict((np.zeros_like(i0[b:b+1]), np.zeros_like(i0[b:b+1]), np.zeros_like(i0[b:b+1]), frames))
    pred = pred[1]
    pred = np.transpose(pred, [0, 3, 1, 2])
    count = np.count_nonzero(np.isnan(pred))

    if count > 0:
        print(f"nan: {count} batch: {b}")
    #tifffile.imwrite(f"/data2/MotherMachineSpots/pred{b}.tiff", data=pred)