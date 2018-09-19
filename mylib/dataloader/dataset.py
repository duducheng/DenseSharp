from collections.abc import Sequence
import random
import os

import numpy as np
from mylib.dataloader.path_manager import PATH
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple

INFO = PATH.info

LABEL = ['AAH', 'AIS', 'MIA', 'IAC']


class ClfDataset(Sequence):
    def __init__(self, crop_size=32, move=3, subset=[0, 1, 2, 3],
                 define_label=lambda l: [l[0] + l[1], l[2], l[3]]):
        '''The classification-only dataset.

        :param crop_size: the input size
        :param move: the random move
        :param subset: choose which subset to use
        :param define_label: how to define the label. default: for 3-output classification one hot encoding.
        '''
        index = []
        for sset in subset:
            index += list(INFO[INFO['subset'] == sset].index)
        self.index = tuple(sorted(index))  # the index in the info
        self.label = np.array([[label == s for label in LABEL] for s in INFO.loc[self.index, 'diagnosis']])
        self.transform = Transform(crop_size, move)
        self.define_label = define_label

    def __getitem__(self, item):
        name = INFO.loc[self.index[item], 'name']
        with np.load(os.path.join(PATH.nodule_path, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'])
        label = self.label[item]
        return voxel, self.define_label(label)

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


class ClfSegDataset(ClfDataset):
    '''Classification and segmentation dataset.'''

    def __getitem__(self, item):
        name = INFO.loc[self.index[item], 'name']
        with np.load(os.path.join(PATH.nodule_path, '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
            # voxel = self.transform(npz['voxel'] * (npz['seg'] * 0.8 + 0.2))
        label = self.label[item]
        return voxel, (self.define_label(label), seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}


def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def get_balanced_loader(dataset, batch_sizes):
    assert len(batch_sizes) == len(LABEL)
    total_size = len(dataset)
    print('Size', total_size)
    index_generators = []
    for l_idx in range(len(batch_sizes)):
        # this must be list, or `l_idx` will not be eval
        iterator = [i for i in range(total_size) if dataset.label[i, l_idx]]
        index_generators.append(shuffle_iterator(iterator))
    while True:
        data = []
        for i, batch_size in enumerate(batch_sizes):
            generator = index_generators[i]
            for _ in range(batch_size):
                idx = next(generator)
                data.append(dataset[idx])
        yield dataset._collate_fn(data)


class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)
