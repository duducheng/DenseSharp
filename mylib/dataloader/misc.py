import pandas as pd

from .path_manager import PATH


def parse_info1():
    ret = pd.DataFrame()
    df = pd.read_excel(PATH.raw_info)
    df = df.loc[df['备注'].isnull()]
    ret['name'] = df['patient'].map(lambda s: "f" + str(s))
    ret['gender'] = df['gender']
    ret['age'] = df['age']
    ret['location'] = df['location']
    ret['diagnosis'] = df['Pathological diagnosis']
    print(ret['gender'].unique())
    print(ret['diagnosis'].unique())
    print(ret['location'].unique())
    print(ret.isnull().sum())
    print(ret.info())
    return ret


def parse_info2():
    ret = pd.DataFrame()
    df = pd.read_excel(PATH.raw2_info)
    df = df.loc[df['备注'].isnull()]
    ret['name'] = df['name'].map(lambda s: "s" + str(s))
    ret['gender'] = df['sex']
    ret['age'] = df['age']
    ret['location'] = df['locaton']
    ret['diagnosis'] = df['pthologic']
    print(ret['gender'].unique())
    print(ret['diagnosis'].unique())
    print(ret['location'].unique())
    print(ret.isnull().sum())
    print(ret.info())
    return ret


def parse_info3():
    ret = pd.DataFrame()
    df = pd.read_excel(PATH.raw3_info)
    df = df.loc[df['备注'].isnull()]
    ret['name'] = df['patients'].map(lambda s: "t" + str(s))
    ret['gender'] = df['sex'].map(lambda s: "M" if s == 1 else "F")
    ret['age'] = df['age']
    ret['location'] = df['location']
    ret['diagnosis'] = df['pathological']
    print(ret['gender'].unique())
    print(ret['diagnosis'].unique())
    print(ret['location'].unique())
    print(ret.isnull().sum())
    print(ret.info())
    return ret


def parse_raw_info():
    info1 = parse_info1()
    info2 = parse_info2()
    info3 = parse_info3()
    raw_info = pd.concat([info1, info2, info3]).reset_index(drop=True)
    print(raw_info.groupby('diagnosis').count())
    return raw_info


def parse_info():
    return pd.read_csv(PATH.case_info)
