import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import _init_paths
from mylib.environ import PATH
from mylib.dataloader.case import Raw, Case
from mylib.utils.lung_utils import preprocess_voxel_seg
from mylib.utils.multicore import TaskRunner
from mylib.utils.misc import split_validation


def parse_raw_info():
    df = pd.read_csv(PATH.raw_info)
    df = df[df['damage'].isnull()]  # damage flag: "yes"
    info = pd.DataFrame()
    info['name'] = df["Folder number"].map(lambda x: "egfr" + str(x))
    info['folder'] = df['Folder name']
    info['invasion_id'] = df['prefolder']
    info['sex'] = df['sex']
    info['age'] = df['age']
    info['location'] = df['location']
    info['pathology'] = df['Pathological：1=AAH,2=AIS,3=MIA,5=IAC'].map(lambda x: x.strip())
    info['EGFR_sub'] = df['EGFR-sub']
    info['EGFR'] = df['EGFR:1=+,2=-'].map(lambda x: 0 if x == 2 else 1)

    info['location'] = info['location'].map(lambda x: "RL" if str(x) == '3' else x)
    info['location'] = info['location'].map(lambda x: "LU" if str(x) == '4' else x)
    info['pathology'] = info['pathology'].map(lambda x: 'IA' if (x == 'IAC' or x == 'IIAC') else x)

    return info


if __name__ == '__main__':
    info = parse_raw_info()

    if not os.path.exists(PATH.case_path):
        os.makedirs(PATH.case_path)

    info.to_csv(PATH.step1_info, index=None)
    print("step1 info file done.")


    def preprocessing(idx):
        raw_info = info
        name = raw_info.loc[idx, 'name']
        raw = Raw(raw_info.loc[idx, 'folder'])
        raw_voxel = raw.get_voxel()
        raw_seg = raw.get_seg()
        v_max = raw_voxel.max()
        v_min = raw_voxel.min()
        voxel, seg = preprocess_voxel_seg(voxel=raw_voxel, seg=raw_seg, spacing=raw.spacing,
                                          window_low=-1024, window_high=400,
                                          new_spacing=[1., 1., 1.], cast_dtype=np.uint8, smooth=1)
        assert voxel.shape == seg.shape
        with open(PATH.get_case(name), 'wb') as f:
            np.savez_compressed(f, voxel=voxel, seg=seg)
        volume = seg.sum()
        return raw.spacing, raw.shape, v_max, v_min, volume, voxel.shape


    args = list(info.index)
    runner = TaskRunner(preprocessing, args, max_workers=8)
    runner.run()

    if runner.errors_:
        # import pdb;pdb.set_trace()
        from IPython import embed

        embed(header="Using IPython environments for debug.")
    else:
        print("no error.")

    # get step2_info
    sorted_results = [r for r in runner.results_ if r[0] == 'success']
    sorted_results.sort(key=lambda t: t[1])

    info['spacingZ'] = [r[2][0][0] for r in sorted_results]
    info['spacingY'] = [r[2][0][1] for r in sorted_results]
    info['spacingX'] = [r[2][0][2] for r in sorted_results]
    info['originD'] = [r[2][1][0] for r in sorted_results]
    info['originH'] = [r[2][1][1] for r in sorted_results]
    info['originW'] = [r[2][1][2] for r in sorted_results]
    info['D'] = [r[2][-1][0] for r in sorted_results]
    info['H'] = [r[2][-1][1] for r in sorted_results]
    info['W'] = [r[2][-1][2] for r in sorted_results]
    info['v_max'] = [r[2][2] for r in sorted_results]
    info['v_min'] = [r[2][3] for r in sorted_results]
    info['volume'] = [r[2][4] for r in sorted_results]

    info.to_csv(PATH.step2_info, index=None)
    print("step2 info file done.")

    info = split_validation(info, 5, by='EGFR')
    info.to_csv(PATH.info, index=None)
    print("info file done.")

    SIZE = 100
    for name in tqdm(info['name']):
        case = Case(name)
        v, s = case.crop(SIZE)
        if s.sum() != case.seg.sum():
            print("not enough size for:", name)
        with open(PATH.get_nodule(name), 'wb') as f:
            np.savez(f, voxel=v, seg=s)
        del case

    print("nodule files saved.")
