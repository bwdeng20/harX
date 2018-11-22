import re
import glob
from torch.utils.data import Dataset
from utils.ParseTraces import parseTrace
import numpy as np
import scipy.io as scio
import sys
from tqdm import tqdm
import os

'''
ubicomp13, a walking detection and step counting data set on smart phones(3A3G3M), around 100hz
https://www.repository.cam.ac.uk/handle/1810/266947
'''

# people, experiment id, carrying location abbreviated as pe_cl
# Notice: regular compression can include at most 2 '.' to match '_' in strings like
# "p3.2Trousers_back_pocket    (3639 8890)"
pe = re.compile(r"p[0-9]+\.[0-9]+")
pe_cl = re.compile(r"p[0-9]+\.[0-9]+[A-Z][a-z]+.[a-z]+.[a-z]+")

mark = re.compile(r'\d+\s\d+')  # start and end mark sample serial number
re_filters = (pe_cl, mark)


def gt_sc(file):
    with open(file) as f:
        recording = f.readlines()
    recording = [p_e_sc.strip('\n').split(' ') for p_e_sc in recording]
    return {p_e_sc[0]: int(p_e_sc[1]) for p_e_sc in recording}


def gt_wd(file):
    """

    :param file:    the abstract file path of walking detection ground truth.
    :return:  a dictionary wherein
                key:  the people id + experiment id + carrying location
                    (e.g, 'p3.1Hand_Held means' the first experiment executed by the third participant
                          with the smart phone held by hand. )

                value:  a list [int: start sample, int: end sample]
    """
    with open(file) as f:
        recording = f.readlines()
    recording = recording[2:]  # the first two lines are irrelevant to the recordings

    pe_cls = [re.search(pe_cl, strip).group() for strip in recording]
    marks = [re.search(mark, strip).group() for strip in recording]

    float_marks = []
    for stamp in marks:
        start, end = stamp.split(' ')[0], stamp.split(' ')[1]
        float_marks.append([int(start), int(end)])
    pe_cls_marks = zip(pe_cls, float_marks)
    return dict(pe_cls_marks)


def gt(gt_sc_file, gt_wd_file):
    """

    :param gt_sc_file: the abstract path of step counting ground truth
    :param gt_wd_file: the abstract path of walking detection ground truth
    :return:a dictionary wherein
            key:    the people id + experiment id + carrying location
                    (e.g, p3.1Hand_Held means the first experiment executed by the third participant
                          with the smart phone held by hand. )
            value:  a list consists of
                    [int: start sample, int: end sample, int: step count]
    """
    try:
        sc = gt_sc(gt_sc_file)
        wd = gt_wd(gt_wd_file)
    except IOError as err:
        raise err

    for wd_key in wd:
        sc_key = re.search(pe, wd_key).group()
        try:
            wd[wd_key].append(sc[sc_key])
        except KeyError:
            print("Step count for %s is missing" % sc_key)
            wd[wd_key].append(None)
            continue
    return wd


def split_by_1st_letter(string):
    i = 0
    letters = 0
    while True:
        char = string[i]
        if 96 < ord(char) < 123 or 64 < ord(char) < 91:
            letters += 1
        if letters == 2:
            return string[:i], string[i:]
        i += 1


class WalkDetectDataset(Dataset):
    def __init__(self, dir="/home/bowendeng/WalkingTrajectoryEstimation/ubicomp13"):
        self.dir = dir
        gt_sc_path = dir + "/groundtruth_SC.txt"
        gt_wd_path = dir + "/groundtruth_WD.txt"
        self.gt = gt(gt_sc_path, gt_wd_path)
        self.keys = list(self.gt.keys())
        out = glob.glob(dir + "/*.out")
        dat = glob.glob(dir + "/*.dat")
        self.seqs = out + dat

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        key = self.keys[idx]
        pe, cl = split_by_1st_letter(key)
        for seq in self.seqs:
            if (pe in seq) and (cl in seq):
                return (key, self.gt[key]), parseTrace(seq)


'''
# USC-HAD, a sensor-based(3A3G) activity recognition data set

'''


def parse_hadmat(matpath, Fs=100):
    d = scio.loadmat(matpath)
    info_keys = ['subject', 'age', 'height', 'weight', 'activity', 'trial']
    info = {key: d[key] for key in info_keys}
    info['Fs'] = 100

    data6d = d['sensor_readings']
    L = len(data6d)
    T = 1 / Fs
    ts = np.arange(L) * T
    accs = data6d[:, :3]
    gyros = data6d[:, 3:]
    data = {'sensor_readings': (ts, accs, gyros), 'activity_number': d['activity_number']}
    return info, data


class USCHAD(Dataset):
    """

    """

    def __init__(self, dir=r'D:\DataSets\Activity_Recognition\SensorBased\USC-HAD\USC-HAD'):
        try:
            self.readings = glob.glob(dir + r"\Subject?\*.mat")
        except Exception as err:
            raise err

    def __len__(self):
        return len(self.readings)

    def __getitem__(self, idx):
        return parse_hadmat(self.readings[idx])


'''
# UCI_HAR, a sensor-based(3A3G) activity recognition data set on smart phone
'''


def avg_split(string, interval):
    if len(string) % interval != 0:
        raise ValueError("String'length should be integral multiple of Interval")

    return [float(string[i:i + interval]) for i in range(0, len(string), interval)]


def parse_hartxt(signal_path):
    with open(signal_path, 'r') as f:
        readings = [row.strip("\n") for row in f.readlines()]

    return [avg_split(row, 16) for row in readings]


def get_signals(signal_dir):
    files = glob.glob(signal_dir + '/*.txt')
    split_symbol = '\\' if 'win' in sys.platform else '/'
    d = {}
    for file in tqdm(files):
        d[file.split(split_symbol)[-1][:-4]] = parse_hartxt(file)
    return d


def rm_train_or_test(string: str):
    return string.replace("_train", "").replace("_test", "")


class UCIHAR(Dataset):
    def __init__(self, dir=
    r'D:\DataSets\Activity_Recognition\SensorBased\UCI HAR Dataset\UCI HAR Dataset\train', transform=None):
        signal_dir = dir + r'\Inertial Signals'

        label_path = glob.glob(dir + '/' + "y_*.txt")[0]

        with open(os.path.dirname(dir) + r'\activity_labels.txt', 'r') as f:
            NoAcs = [line.strip().split(" ") for line in f.readlines()]
            self.activity = {int(NoAc[0]): NoAc[1] for NoAc in NoAcs}

        print("Loading sensor readings into RAM ...")
        self.signals = get_signals(signal_dir)

        # if no list(), some keys wont be converted for unknown reason!!!
        # if you know, please contact me. E-mail: dbwtimteo@outlook.com
        for key in list(self.signals.keys()):
            self.signals[rm_train_or_test(key)] = self.signals.pop(key)

        with open(label_path, 'r') as f:
            self.labels = [int(row.strip().replace("\n", '')) for row in f.readlines()]

        assert len(self.labels) == len(self.signals['body_acc_x'])

        # samples(for train set 7352 and 2947 for test set) * sensors(9) * window_size(128)
        self.tensor_signals = np.stack(self.signals.values(), axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {"input": self.tensor_signals[idx], "label": self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Heterogeneous(Dataset):
    def __init__(self, dir=''):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class StrideLengthDataset(Dataset):
    def __init__(self, dir=""):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def norm3d(data3d):
    return [np.sqrt(x * x + y * y + z * z) for x, y, z in data3d]


if __name__ == "__main__":
    # test function gt_sc
    groundtruth_sc = r'D:\WalkingTrajectoryEstimation\ubicomp13\groundtruth_SC.txt'
    sc = gt_sc(groundtruth_sc)
    print(sc['p3.1'])
    # test function gt_wd
    groundtruth_wd = r'D:\WalkingTrajectoryEstimation\ubicomp13\groundtruth_WD.txt'
    wd = gt_wd(groundtruth_wd)
    print(wd['p3.1Hand_held'])
    # test function gt
    gt_dict = gt(groundtruth_sc, groundtruth_wd)
    print(gt_dict['p3.1Hand_held'])
    for key in gt_dict:
        print(key)

    # test class WalkDetectDataset
    dataset = WalkDetectDataset(dir=r'D:\WalkingTrajectoryEstimation\ubicomp13')
    info, seqs = dataset[10]
    print(info)
    print("==> WalkDetectDataset Class Ready!")

    # test class USCHAD
    had = USCHAD()
    info_dict, data_dict = had[100]
    print(info_dict)
    print(data_dict.keys())

    print("==> USCHAD Class Ready!")

    # test class UCIHAR
    ucihar = UCIHAR()
    reading, label = ucihar[10]["input"], ucihar[10]["label"]
    print(label, reading.shape)
    print(ucihar.activity)

    print("==> UCIHAR Class Ready!")
