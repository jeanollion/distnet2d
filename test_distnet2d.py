import time
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter
path_root = Path(__file__).parents[1]
sys.path.append(path_root.joinpath("dataset_iterator").__str__())
sys.path.append(path_root.joinpath("distne2d").__str__())
#print(f"root={path_root} pixmclass {path_root.joinpath('pix_mclass')}")
import tensorflow as tf


import numpy as np
from dataset_iterator.tile_utils import extract_tiles
#a = np.zeros(shape=(10, 256, 250, 2))
#tiles = extract_tiles(a, tile_shape=(192, 192), n_tiles=17)
#print(f"tiles: {tiles.shape}")

#path = "/data/DL/DistNet2D/Maxime/ds_isolement2.h5"
from dataset_iterator import extract_tile_random_zoom_function
#path = "/data/DL/DistNet2D/IntermediateChannels/train8s_sub8_12_16.h5"
#path = "/data/DL/DiSTNet2D/PhC_C2DH_PA14.h5"
path = "/data/Images/TestFluo/test.h5"
from distnet_2d.data import DyDxIterator
from distnet_2d.data.dydx_iterator import CHANNEL_KEYWORDS, ARRAY_KEYWORDS
from dataset_iterator.image_data_generator import IlluminationImageGenerator, ScalingImageGenerator, ImageGeneratorList, get_image_data_generator
from dataset_iterator.datasetIO import get_datasetIO, MemoryIO
from distnet_2d.model import get_distnet_2d, architectures

dn = get_distnet_2d((None, None, 4), 5, True, category_number=2, config=architectures.BlendD3(filters=128, self_attention=0, attention=0), predict_edm_derivatives=False, predict_gcdm_derivatives=False)

#tf.keras.utils.plot_model(dn, "/data/model.png", show_shapes=True)
#dn.load_weights("/data/DL/DistNet2D/MotherMachinePhase/distnet2d_mm_phase_D3ASA16_5.h5")
#print(dn.summary())


data_gen = ScalingImageGenerator("RANDOM_CENTILES", dataset=path, channel_name="raw", per_image=True, verbose=True)

affine_transform_parameters = {}
data_generator = get_image_data_generator(scaling_parameters={'mode': "RANDOM_CENTILES", "min_centile": 0.1, "max_centile":99.9}, affine_transform_parameters=affine_transform_parameters)
affine_transform_parameters_mask = None if affine_transform_parameters is None else {**affine_transform_parameters, "interpolation_order": 0}
mask_generator = get_image_data_generator(scaling_parameters=[], affine_transform_parameters=affine_transform_parameters_mask)

tiling_parameters = {"tile_shape":(128, 128), "n_tiles":1, "interpolation_order":1,"perform_augmentation":True,"augmentation_rotate":"true","zoom_range":[0.8333333333333334,1.2],"aspect_ratio_range":[0.8333333333333334,1.2],"random_stride":True,"random_channel_jitter_shape":[10,10]}
it = DyDxIterator(dataset=path, group_keyword=None, frame_window=5, erase_edge_cell_size=50, center_mode="MEDOID",
                        center_distance_mode="EUCLIDEAN",
                        channel_keywords=CHANNEL_KEYWORDS.copy() + ['/Fluo'],
                        input_label_keywords=["/Bacteria"],
                        array_keywords=ARRAY_KEYWORDS,
                        elasticdeform_parameters={},
                        return_edm_derivatives=False,
                        image_data_generators=[data_generator, mask_generator, data_generator],
                        batch_size=1, step_number=0, extract_tile_function= extract_tile_random_zoom_function(**tiling_parameters),
                        aug_frame_subsampling=15, verbose=False, shuffle=False, return_image_index=False)


it.disable_random_transforms(True, True)

it.return_central_only = True
#it.return_label_rank = True
it.incomplete_last_batch_mode = 0
print(f"{len(it)}")
t0 = time.time()
x, y = it[0]
t1 = time.time()
print(f"computation time: {t1-t0}")
print(f"shape : x={x.shape} y={[yy.shape for yy in y]}")
fw = 5
n_frames = fw * 2 + 1
n_frame_pairs = fw * 2
n_frame_pairs += (fw - 1) * 2
d_indices = [fw-1, n_frame_pairs+fw]
lm_indices = [d_indices[0]*3 + i for i in range(3)] + [d_indices[1]*3 + i for i in range(3)]


print(f"d_indices: {d_indices} lm_indices: {lm_indices} ")
y_pred = dn.predict(x)
print(f"pred shape: {[yy.shape for yy in y_pred]}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5), ncols=7, nrows=1)
ax[0].imshow(x[0,...,0,fw], interpolation="nearest", aspect='equal', origin="upper")
ax[1].imshow(x[0,..., 1,fw], interpolation="nearest", aspect='equal', origin="upper")
ax[2].imshow(x[0,..., 2,fw], interpolation="nearest", aspect='equal', origin="upper")
ax[3].imshow(x[0,..., 3,fw], interpolation="nearest", aspect='equal', origin="upper")
ax[4].imshow(y[0][0,...,0], interpolation="nearest", aspect='equal', origin="upper")
#ax[5].imshow(y_pred[0][0,...,fw], interpolation="nearest", aspect='equal', origin="upper")
ax[5].imshow(y[1][0,...,0], interpolation="nearest", aspect='equal', origin="upper")
ax[6].imshow(y[5][0,...,0], interpolation="nearest", aspect='equal', origin="upper")
#ax[4].imshow(y_pred[1][0,...,fw], interpolation="nearest", aspect='equal', origin="upper")
#ax[5].imshow(y[2][0,...,0], interpolation="nearest", aspect='equal', origin="upper")
#ax[6].imshow(y_pred[2][0,...,d_indices[0]], interpolation="nearest", aspect='equal', origin="upper")
#ax[7].imshow(y[2][0,...,1], interpolation="nearest", aspect='equal', origin="upper")
#ax[8].imshow(y_pred[2][0,...,d_indices[1]], interpolation="nearest", aspect='equal', origin="upper")
#ax[9].imshow(y[4][0,...,0], interpolation="nearest", aspect='equal', origin="upper")
#ax[10].imshow(np.argmax(y_pred[4][0,...,lm_indices[0]:lm_indices[2]+1], axis=-1), interpolation="nearest", aspect='equal', origin="upper", cmap='gray', vmin=0, vmax=2)
#ax[11].imshow(y[4][0,...,1], interpolation="nearest", aspect='equal', origin="upper")
#ax[12].imshow(np.argmax(y_pred[4][0,...,lm_indices[-3]:lm_indices[-1]+1], axis=-1), interpolation="nearest", aspect='equal', origin="upper", cmap='gray', vmin=0, vmax=2)
#ax[13].imshow(y[-3][0,...,0], interpolation="nearest", aspect='equal', origin="upper")
plt.show()


#print(f"edm equals: {np.array_equal(y[0][..., 3:4], y2[0])}")
#print(f"center equals: {np.array_equal(y[1][..., 3:4], y2[1])}")
#print(f"dy equals: {np.allclose(y[2][..., d_indices], y2[2])}")
#print(f"dx equals: {np.allclose(y[3][..., d_indices], y2[3])}")
#print(f"lm equals: {np.allclose(y[4][..., d_indices], y2[4])}")

