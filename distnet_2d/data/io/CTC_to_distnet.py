from os import listdir
from os.path import isfile, isdir, join, basename
import h5py
from tifffile import imread
import numpy as np
from scipy.ndimage import find_objects
from .processing import _close_and_fill

# list folders without underscore
# - 01 -> raw tFFF.tif
# - 01_ST/SEG -> segmentation man_segFFF.tif
# - 01_GT/TRA/man_track.txt -> tracking: LABEL | START_FRAME | END_FRAME | PARENT_LABEL
# - 01_GT/TRA/man_trackFFF.tif
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

# ds_dirs = [join(main_dir, f) for f in listdir(main_dir) if isdir(join(main_dir, f))]

def _get_label(masked_label_im):
    ob_l,counts = np.unique(masked_label_im, return_counts=True)
    if len(ob_l)>1 and ob_l[0]==0:
        ob_l = ob_l[1:]
        counts = counts[1:]
    if len(ob_l)>1:
        return ob_l[counts.argmax()]
    else:
        return ob_l[0]

def convert_to_distnet(*dataset_dirs, closing_iteration:int = 1):
    for dir in dataset_dirs:
        print(f"Dataset dir: {dir}")
        dirs = [f for f in listdir(dir) if isdir(join(dir, f)) and '_' not in f]
        out_dir = join(dir, basename(dir)+".h5")
        with h5py.File(out_dir, 'w') as out_file:
            for position in dirs:
                print(f"Position: {position}")
                raw_dir = join(dir, position)
                seg_dir = join(dir, position+"_ST", "SEG")
                seg_dir_gt = join(dir, position+"_GT", "SEG")
                track_dir = join(dir, position+"_GT", "TRA")
                # parse tracks
                track_file = join(track_dir, "man_track.txt")
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
                    label_file_GT = join(seg_dir_gt, f"man_seg{frame_s}.tif")
                    label_file_ST = join(seg_dir, f"man_seg{frame_s}.tif")
                    if not isfile(label_file_GT) and not isfile(label_file_ST):
                        label_im = np.zeros(raw_ims.shape[1:], dtype="int16")
                        label_im_alt = None
                    else:
                        if isfile(label_file_GT):
                            label_im = imread(label_file_GT)
                            if isfile(label_file_ST):
                                label_im_alt = imread(label_file_ST)
                            else:
                                label_im_alt = None
                        else:
                            label_im = imread(label_file_ST)
                            label_im_alt = None

                    track_label_file = join(track_dir, f"man_track{frame_s}.tif")
                    if isfile(track_label_file):
                        track_label_im = imread(track_label_file)
                    else:
                        track_label_im = np.zeros(raw_ims.shape[1:], dtype="int16")

                    track_objects = find_objects(track_label_im)#, labels=track_label_im, index = np.arange(1, track_label_im.max()+1))
                    if isinstance(track_objects, tuple):
                        track_objects=[track_objects]
                    target_label_im = np.zeros(raw_ims.shape[1:], dtype="int16")
                    for idx, o in enumerate(track_objects): # check that track label correspond to segmentation label
                        l = idx + 1
                        if o is not None:
                            mask = track_label_im[o] == l
                            ob_l = _get_label(label_im[o][mask])
                            if ob_l==0:
                                if label_im_alt is not None:
                                    ob_l = _get_label(label_im_alt[o][mask])
                                    if ob_l>0: # transfer object from alt label_im
                                        target_label_im[label_im_alt==ob_l] = l
                                        #print(f"Warning: no object at frame: {frame_s} for track label: {l}  : transfered from ST")
                                if ob_l==0:
                                    target_label_im[o][mask] = l
                                    print(f"Warning: no object at frame: {frame_s} for track label: {l}  : track label written")
                            else:
                                target_label_im[label_im==ob_l] = l
                    _close_and_fill(target_label_im, iterations = closing_iteration)
                    ds_labels[frame] = target_label_im
                    objects = find_objects(target_label_im)
                    if isinstance(objects, tuple):
                        objects=[objects]
                    for idx, o in enumerate(objects):
                        l = idx + 1
                        if o is not None:
                            track = tracks.get(l, None)
                            if track is not None:
                                prev_l = track[1] if track[0]==frame else l
                                if prev_l>0:
                                    mask = target_label_im[o] == l
                                    prev_labels[frame][o][mask] = prev_l
                                    #print(f"position: {position} frame: {frame} label: {l} prev label: {prev_l}")
                            else:
                                print(f"Warning position: {position} frame: {frame} label: {l} slice:{slice} has no track")
                out_file.create_dataset(f"{position}/prevRegionLabels", data=prev_labels)
        print("Dataset processed")
