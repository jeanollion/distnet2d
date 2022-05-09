from os import listdir
from os.path import isfile, isdir, join, basename
import h5py
from scipy.ndimage import find_objects
from tifffile import imread
import numpy as np
main_dir = "/data/Images/CellTrackingChallenge"

# list folders without underscore
# - 01 -> raw tFFF.tif
# - 01_ST/SEG -> segmentation man_segFFF.tif
# - 01_GT/TRA/man_track.txt -> tracking: LABEL | START_FRAME | END_FRAME | PARENT_LABEL

# compute previous labels:
# - parse track list -> dict LABEL -> {START_FRAME, END_FRAME, PARENT_LABEL}
# - for each image t:
#     for each label:
#       if t>0 && t==START_FRAME:
#          PREVIOUS_LABEL = PRENT_LABEL
#       else:
#          PREVIOUS_LABEL = LABEL
# output:
# - 01 / raw -> raw images
# - 01 / labels -> 01_ST/SEG
# - 01 / previousLabels -> previous labels

ds_dirs = [join(main_dir, f) for f in listdir(main_dir) if isdir(join(main_dir, f))]
for dir in ds_dirs:
    dirs = [f for f in listdir(dir) if isdir(join(dir, f)) and '_' not in f]
    out_dir = join(dir, basename(dir)+".h5")
    with h5py.File(out_dir, 'w') as out_file:
        for position in dirs:
            raw_dir = join(dir, position)
            seg_dir = join(dir, position+"_ST", "SEG")
            # parse tracks
            track_file = join(dir, position+"_GT", "TRA", "man_track.txt")
            tracks = dict()
            with open(track_file) as tf:
                for t in tf.readlines():
                    tt = t.split()
                    tracks[int(tt[0])] = [int(tt[1]), int(tt[3])]
            print(f"parsed tracks: {tracks}")
            # compute previous label
            images = [f for f in listdir(raw_dir) if f.startswith("t") and f.endswith(".tif")]
            images.sort()
            raw_ims = np.array([imread(join(raw_dir, f)) for f in images])
            print(f"raw im shape: {raw_ims.shape}")
            out_file.create_dataset(f"{position}/raw", data=raw_ims)
            ds_labels = out_file.create_dataset(f"{position}/regionLabels", raw_ims.shape, dtype='int16')
            prev_labels = np.zeros(raw_ims.shape, dtype="int16")
            end_digit_idx = images[0].index(".tif")
            for im_name in images:
                frame_s = im_name[1:end_digit_idx]
                frame = int(frame_s)
                label_im = imread(join(seg_dir, f"man_seg{frame_s}.tif"))
                ds_labels[frame] = label_im
                objects = find_objects(label_im)
                for idx, o in enumerate(objects):
                    l = idx + 1
                    if o is not None:
                        track = tracks.get(l, None)
                        if track is not None:
                            prev_l = track[1] if track[0]==frame else l
                            if prev_l>0:
                                mask = label_im[o] == l
                                prev_labels[frame][o][mask] = prev_l
                                #print(f"position: {position} frame: {frame} label: {l} prev label: {prev_l}")
                        else:
                            print(f"position: {position} frame: {frame} label: {l} slice:{slice} has no track")
            out_file.create_dataset(f"{position}/prevRegionLabels", data=prev_labels)
