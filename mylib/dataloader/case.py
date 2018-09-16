import os

import nibabel as nib
import nrrd
import numpy as np
from skimage.morphology import binary_closing, ball

from .path_manager import PATH
from ..utils.misc import resize, window_clip, crop_at_zyx_with_dhw, _triple

if PATH is None:
    raise OSError("Setup `os.environ` first.")


class Case:
    def __init__(self, patient):
        with np.load(PATH.get_patient_case(patient)) as npz:
            self.voxel = npz['voxel']
            self.seg = npz['seg']

    def crop(self, size=80, zyx=None, fill_with=0):
        size = _triple(size)
        if zyx is None:
            zyx = self.center
        crop_v = crop_at_zyx_with_dhw(self.voxel, zyx, size, fill_with)
        crop_s = crop_at_zyx_with_dhw(self.seg, zyx, size, fill_with)
        return crop_v, crop_s

    @property
    def center(self):
        z, y, x = np.where(self.seg)
        return z.mean(), y.mean(), x.mean()


class Raw:
    def __init__(self, patient,extra=False):
        if extra:
            self.base = patient
        else:
            self.base = PATH.get_patient_folder(patient)
        files = os.listdir(self.base)
        self.voxel_file = self._find(files,  # "label" not included
                                     rule=lambda f: "label" not in f and (f.endswith(".nii.gz")
                                                                          or f.endswith(".nii")
                                                                          or f.endswith(".nrrd")))
        self.seg_file = self._find(files,  # "label" included
                                   rule=lambda f: "label" in f)
        self.spacing = None
        self.shape = None

    @property
    def voxel(self):
        arr = self._parse(self.voxel_file)
        return arr

    @property
    def seg(self):
        arr = self._parse(self.seg_file)
        assert arr.sum() != 0, "Invalid %s" % self.seg_file
        return arr

    def preprocess(self, window_low=-1024, window_high=400,
                   new_spacing=[1., 1., 1.], cast_dtype=np.uint8, smooth=1):
        voxel = self.voxel
        seg = self.seg
        v_max = voxel.max()
        v_min = voxel.min()
        resized_voxel, _ = resize(voxel, self.spacing, new_spacing)
        resized_seg, _ = resize(seg, self.spacing, new_spacing)
        mapped_voxel = window_clip(resized_voxel, window_low, window_high, dtype=cast_dtype)
        smoothed_seg = binary_closing(resized_seg, ball(smooth))
        return mapped_voxel, smoothed_seg, v_max, v_min

    def _find(self, files, rule):
        candidates = []
        for f in files:
            if rule(f):
                candidates.append(os.path.join(self.base, f))
        latest = max(candidates, key=lambda x: os.stat(x).st_mtime)  # the latest modified
        return latest

    def _parse(self, path):
        if path.endswith('.nrrd'):
            arr, opts = nrrd.read(path)
            affine = np.array([[abs(float(c)) for c in r] for r in opts['space directions']])
            orient = list(reversed(affine.argmax(axis=1)))
            spacing = [affine[r][c] for r, c in zip([2, 1, 0], orient)]
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            nii = nib.load(path)
            affine = nii.affine
            orient = list(reversed(np.abs(affine[:3, :3]).argmax(axis=1)))
            spacing = [abs(x) for x in affine[[2, 1, 0], orient]]  # expect: [Z,Y,X]
            # origin = list(affine[[2, 1, 0], [-1, -1, -1]])
            arr = nii.get_data()
        else:
            raise ValueError("Invalid %s" % path)
        arr = arr.transpose(*orient)
        shape = arr.shape
        if self.shape is None:
            assert shape[1:] == (512, 512), "Invalid %s" % path
            self.shape = shape
        else:
            assert self.shape == shape, "Invalid %s" % path
        if self.spacing is None:
            self.spacing = spacing
        else:
            assert np.allclose(self.spacing, spacing), "Invalid %s" % path
        return arr
