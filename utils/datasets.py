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
Dataset proposed in this paper
-----------------------------------------------------------------------------
Walk Detection and Step Counting on Unconstrained Smartphones
------------------------------------------------------------------------------
    
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
# INFO: 100hz, totally separated reading windows for 12 kinds of activities
1. Walking Forward
2. Walking Left
3. Walking Right
4. Walking Upstairs
5. Walking Downstairs
6. Running Forward
7. Jumping Up
8. Sitting
9. Standing
10. Sleeping
11. Elevator Up
12. Elevator Down
'''


def parse_hadmat(matpath, Fs=100):
    d = scio.loadmat(matpath)
    INFO_keys = ['subject', 'age', 'height', 'weight', 'activity', 'trial']
    INFO = {key: d[key] for key in INFO_keys}
    INFO['Fs'] = 100
    d['sensor_readings'] = np.float32(d['sensor_readings'])
    data6d = d['sensor_readings']
    L = len(data6d)
    T = 1 / Fs
    ts = np.arange(L) * T
    accs = data6d[:, :3]
    gyros = data6d[:, 3:]
    data = {'input': (ts, accs, gyros), 'label': d['activity_number']}
    return INFO, data


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
# UCI_HAR, a sensor-based(3A3G) activity recognition data set collected by smart phones attached to the waist 
# INFO: 50hz, 50% overlap in each window.
#       2.56 sec and 50% overlap (128 readings/window)
# Archive:  https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
# NOTICE:   Though the raw data were collected by 3-axial accelerator and 3-axial gyroscope, there are 
            totally 9 dimension in each sample point for the organizers have processed the sensor 
            readings with ButterWorth low-pass filter to separate the acceleration caused by gravity and that 
            originating from body locomotion.
        
            U can use 9-D sensor readings or just 3-axial Acc plus 3-axial Gyro which weren't processed using
            a ButterWorth low-pass filter.
'''


def avg_split(string, interval):
    if len(string) % interval != 0:
        raise ValueError("String'length should be integral multiple of Interval")

    # float(.) return 64 bits float number which is not necessary so we utilize np.float32() to reduce memory cost
    return [np.float32(float(string[i:i + interval])) for i in range(0, len(string), interval)]


def parse_hartxt(signal_path):
    with open(signal_path, 'r') as f:
        readings = [row.strip("\n") for row in f.readlines()]
    # since every 16 characters(including symbol and value) represent one number
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
    def __init__(self, dir=r'D:\DataSets\Activity_Recognition\SensorBased\UCI HAR Dataset\UCI HAR Dataset\train',
                 transform=None):
        signal_dir = dir + r'\Inertial Signals'

        label_path = glob.glob(dir + '/' + "y_*.txt")[0]

        # parse the activity and its corresponding number,e.g, 2: 'WALKING_UPSTAIRS'.
        with open(os.path.dirname(dir) + r'\activity_labels.txt', 'r') as f:
            NoAcs = [line.strip().split(" ") for line in f.readlines()]
            self.activity = {int(NoAc[0]): NoAc[1] for NoAc in NoAcs}

        print("\nLoading sensor readings into RAM ...\n")
        self.signals = get_signals(signal_dir)

        # if no list(), some keys wont be converted for unknown reason!!!
        # if you know, please tell me. E-mail: dbwtimteo@outlook.com
        for key in list(self.signals.keys()):
            self.signals[rm_train_or_test(key)] = self.signals.pop(key)

        with open(label_path, 'r') as f:
            self.labels = [int(row.strip().replace("\n", '')) for row in f.readlines()]

        # check whether the data set is consistent
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


"""
# HAPT Data Set, a sensor-based(3A3G) activity recognition data set on smart phone 
# INFO: 12 kinds of activities
        1 WALKING           
        2 WALKING_UPSTAIRS  
        3 WALKING_DOWNSTAIRS
        4 SITTING           
        5 STANDING          
        6 LAYING            
        7 STAND_TO_SIT      
        8 SIT_TO_STAND      
        9 SIT_TO_LIE        
        10 LIE_TO_SIT        
        11 STAND_TO_LIE      
        12 LIE_TO_STAND      

# Archive: 
# http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
"""


def parse_hapt_labels(path):
    """
    Column 1: experiment number ID,
    Column 2: user number ID,
    Column 3: activity number ID
    Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
    Column 5: Label end point (in number of signal log samples)
    :param path: the abstract file path of "labels.txt" file which consists of 1214 rows.
    :return:
    """
    with open(path, "r") as f:
        lines = f.readlines()
    lines = list(map(lambda s: s.strip("\n").strip(" ").split(" "), lines))
    return np.array(lines, dtype=np.int32)


def segment_idx(st_ets, segment_size):
    segment_sts = map(lambda st_et: np.arange(st_et[0], st_et[1] + 1 - segment_size + 1,
                                              np.int32(segment_size / 2)), st_ets)
    return list(segment_sts)


def find_row_idx(item, rows):
    """
    https://stackoverflow.com/questions/18927475/numpy-array-get-row-index-searching-by-a-row
    :param item: the row whose index u want to search
    :param rows: the stacked rows, i.e, a 2-D array
    :return:
    """
    bool_mat = item == rows
    return np.where(np.all(bool_mat, axis=1))[0]


def extract_basic_features(x, y, z):
    n = x.shape[0]  # the segment size
    # 6 features
    features1 = [np.mean(x), np.mean(y), np.mean(z), np.std(x), np.std(y), np.std(z)]

    # 9 features
    features2 = [np.mean(np.abs(x - x.mean())), np.mean(np.abs(y - y.mean())), np.mean(np.abs(z - z.mean()))]

    # 10 features
    feature3 = [np.mean(np.sqrt(x ** 2 + y ** 2 + z ** 2))]

    # 40 features
    features4 = np.concatenate([np.histogram(x, 10)[0] / n, np.histogram(y, 10)[0] / n, np.histogram(z, 10)[0] / n])
    return np.concatenate([features1, features2, feature3, features4])


def segment(raw_data_dir, des_dir, label_info, segment_size=128, transition=True,
            test_pids=(2, 4, 9, 10, 12, 13, 18, 20, 24),
            fea_size=40):
    e_id_p_ids = label_info[:, :2]
    e_id_p_ids_unique = np.unique(e_id_p_ids, axis=0)

    labels = label_info[:, 2]

    st_ets = label_info[:, -2:]
    st_ets = st_ets.tolist()
    segment_sts = segment_idx(st_ets, segment_size)

    x_train = np.empty((1, segment_size, 6), dtype=np.float64)
    y_train = np.empty(1, dtype=np.int32)
    feas_train = np.empty((1, fea_size), dtype=np.float64)
    x_test = np.empty((1, segment_size, 6), dtype=np.float64)
    y_test = np.empty(1, dtype=np.int32)
    feas_test = np.empty((1, fea_size), dtype=np.float64)

    print(" ==> segmentation launches ... ")
    for e_id, p_id in tqdm(e_id_p_ids_unique):
        sample_idx = find_row_idx([e_id, p_id], e_id_p_ids)
        if e_id < 10:
            e_id = "0" + str(e_id)
        else:
            e_id = str(e_id)
        if p_id < 10:
            p_id = "0" + str(p_id)
        else:
            p_id = str(p_id)
        with open(raw_data_dir + "/acc_exp" + e_id + "_user" + p_id + ".txt") as f:
            acc_xyzs = f.readlines()
        acc_xyzs = list(map(lambda s: s.strip("\n").strip(" ").split(" "), acc_xyzs))
        # length x 3
        acc_xyzs = np.array(acc_xyzs, dtype=np.float)

        with open(raw_data_dir + "/gyro_exp" + e_id + "_user" + p_id + ".txt") as f:
            gyro_xyzs = f.readlines()

        gyro_xyzs = list(map(lambda s: s.strip("\n").strip(" ").split(" "), gyro_xyzs))
        # length x 3
        gyro_xyzs = np.array(gyro_xyzs, dtype=np.float)

        # length x 6
        raw_series = np.concatenate([acc_xyzs, gyro_xyzs], axis=1)

        for idx in sample_idx:
            # (1,)
            label_winidow = np.array([labels[idx]])
            if not transition:  # do not consider the transition activities
                if labels[idx] > 6:  # transition activities have numbers greater than 6(7-12)
                    # check the next idx in sample_idx
                    continue
            for windows_st in segment_sts[idx]:
                # windows_st-1 because the original indexes of column4 and column5 in 'labels.txt' begins from 1 not 0
                # (128,6)
                acc_window = acc_xyzs[windows_st - 1:windows_st - 1 + segment_size, :]
                acc_gyro_window = raw_series[windows_st - 1:windows_st - 1 + segment_size, :]
                # (40,)
                feature_window = extract_basic_features(acc_window[:, 0], acc_window[:, 1], acc_window[:, 2])

                if p_id in test_pids:
                    x_test = np.concatenate([x_test, acc_gyro_window[np.newaxis, :]])
                    feas_test = np.concatenate([feas_test, feature_window[np.newaxis, :]])
                    y_test = np.append(y_test, label_winidow)
                else:
                    x_train = np.concatenate([x_train, acc_gyro_window[np.newaxis, :]])
                    feas_train = np.concatenate([feas_train, feature_window[np.newaxis, :]])
                    y_train = np.append(y_train, label_winidow)

    print(r"\n ===>  saving the segmented data(as np.array) into '.npy' file \n")
    np.save(des_dir + "/x_train.npy", np.delete(x_train, 0, axis=0))
    np.save(des_dir + "/y_train.npy", np.delete(y_train, 0, axis=0))
    np.save(des_dir + "/fea_train.npy", np.delete(feas_train, 0, axis=0))

    np.save(des_dir + "/x_test.npy", np.delete(x_test, 0, axis=0))
    np.save(des_dir + "/y_test.npy", np.delete(y_test, 0, axis=0))
    np.save(des_dir + "/fea_test.npy", np.delete(feas_test, 0, axis=0))
    print(r"\n ===> all segmented data saved\n")

    return None


class HAPT(Dataset):
    def __init__(self, raw_data_dir="D:/DataSets/Activity_Recognition/SensorBased/HAPT Data Set/RawData",
                 des_dir=r"D:/WalkingTrajectoryEstimation/harX/data", train=1, float=32,
                 segment_size=128, transform=None, test_ids=(2, 4, 9, 10, 12, 13, 18, 20, 24),
                 posture_trans=True):
        self.transform = transform

        self.dir = raw_data_dir
        self.s = segment_size
        self.d_dir = des_dir
        self.test_ids = test_ids
        self.transition = posture_trans
        self.label_info = parse_hapt_labels(raw_data_dir + "/labels.txt")

        self.is_segmented = self.check_segmented()

        if not self.is_segmented:  # if there is no segmented data under the destination directory 'des_dir'
            self.check_segment_size()
            try:
                self.segment()
                self.is_segmented = self.check_segmented()
            except Exception as e:
                raise e

        # loading all segmented data
        if train:
            x_pth = des_dir + "/x_train.npy"
            y_pth = des_dir + "/y_train.npy"
            fea_pth = des_dir + "/fea_train.npy"

        else:
            x_pth = des_dir + "/x_test.npy"
            y_pth = des_dir + "/y_test.npy"
            fea_pth = des_dir + "/fea_test.npy"
        self.x = np.load(x_pth)
        self.y = np.load(y_pth)
        self.fea = np.load(fea_pth)
        assert self.x.shape[0] == self.y.shape[0] == self.fea.shape[0]
        self.size = self.x.shape[0]

        if float == 32:
            self.dtype = np.float32
        elif float == 64:
            self.dtype = np.float64

    def check_segment_size(self):
        intervals = self.label_info[:, -1] - self.label_info[:, -2]
        if not np.all(intervals > 0):
            raise ValueError(
                "This segment size {} is larger than some labelled activity sensor reading intervals".format(self.s, ))

    def check_segmented(self):
        # a set
        all_files = {self.d_dir + "\\x_train.npy", self.d_dir + "\\y_train.npy",
                     self.d_dir + "\\fea_train.npy", self.d_dir + "\\x_test.npy", self.d_dir + "\\y_test.npy",
                     self.d_dir + "\\fea_test.npy"}
        # a list
        dir_files = glob.glob(self.d_dir + "/*.npy")

        return all_files == set(dir_files)

    def segment(self):
        segment(raw_data_dir=self.dir, des_dir=self.d_dir, label_info=self.label_info, segment_size=self.s,
                transition=self.transition,
                test_pids=self.test_ids)
        return True

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = {"input": (self.x[idx]).astype(self.dtype),
                  "label": self.y[idx],
                  "fea": (self.fea[idx]).astype(self.dtype)}
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
    # test class HAPT
    hapt = HAPT()
    inp, lab, fea = hapt[10]["input"], hapt[10]["label"], hapt[10]["fea"]
    print(inp.shape, inp.dtype, "\n",
          lab, lab.dtype, "\n", fea.shape, fea.dtype)
    print("==> HAPT Class Ready!")

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
    INFO, seqs = dataset[10]
    print(INFO)
    print(seqs[0][0] - seqs[0][-1])
    print("==> WalkDetectDataset Class Ready!")

    # test class USCHAD
    had = USCHAD()
    INFO_dict, data_dict = had[100]
    print(data_dict["input"])
    print(INFO_dict)
    print(data_dict.keys())

    print("==> USCHAD Class Ready!")

    # test class UCIHAR
    ucihar = UCIHAR()
    reading, label = ucihar[10]["input"], ucihar[10]["label"]
    print(label, "reading.shape: ", reading.shape, "/n reading.dtype:", reading.dtype)
    print(ucihar.activity)

    print("==> UCIHAR Class Ready!")
