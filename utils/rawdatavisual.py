import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft


def oddata(sensor_data, dim):
    """
    :param sensor_data: three dimensional sensor data
    :param dim: Along which dimension you extract the data
    :return: sensor data in one dimension(x or y or z)
    """
    return [elem[dim] for elem in sensor_data]


def plot_1trace(data, gt_info=None):
    """
     Depict 4 figures displaying raw data of accelerator, gyroscope, and geomagnetic field sensors.
    These figures are in the order of 3-dimensional accelerator data,3-dimensional gyroscope data,
    3-dimensional geomagnetic field data, and accelerator norm data.
    NOTICE:three dimension coordinates refer to the Sensor Coordinate System illustrated at
    https://developer.android.com/guide/topics/sensors/sensors_overview#sensors-coords

    :param data:                a tuple or list consists of ts(ts), accData, ts, gyroData, ts, magnData.
    :param gt_info:          a tuple composed by
                                    person_id+experiment_id+ arrying_location, (start, end) and the step count.
                                e.g. ('p1.1Hand_held', [1607, 5393, 70])
    :return:
    """
    ts, accData, gyroData, magnData = data[0], data[1], data[3], data[5]
    if gt_info:
        pel, (start, end, sc) = gt_info

    plt.figure()

    plt.subplot(411)
    handles = []
    for i in range(3):
        acc_1dim, = plt.plot(ts, oddata(accData, i), label='acc_%d' % i)
        handles.append(acc_1dim)

    if gt_info:
        plt.axvline(ts[start], linestyle='dashed', color='black')
        plt.axvline(ts[end], linestyle='dashed', color='black')

    plt.legend(handles)
    plt.title('Accelerator', loc='left')
    # -------------------------------------------
    plt.subplot(412)
    handles = []
    for i in range(3):
        gyro_1dim, = plt.plot(ts, oddata(gyroData, i), label='gyro_%d' % i)
        handles.append(gyro_1dim)

    if gt_info:
        plt.axvline(ts[start], linestyle='dashed', color='black')
        plt.axvline(ts[end], linestyle='dashed', color='black')

    plt.legend(handles)
    plt.title('Gyroscope', loc='left')
    # -------------------------------------------
    plt.subplot(413)
    handles = []
    for i in range(3):
        magn, = plt.plot(ts, oddata(magnData, i), label='magn_%d' % i)
        handles.append(magn)

    if gt_info:
        plt.axvline(ts[start], linestyle='dashed', color='black')
        plt.axvline(ts[end], linestyle='dashed', color='black')

    plt.legend(handles=handles)
    plt.title('Magnetic filed ', loc='left')
    # -------------------------------------------
    plt.subplot(414)
    acc_norm, = plt.plot(ts, [np.sqrt(x * x + y * y + z * z) for x, y, z in accData],
                         label='acc_norm')
    plt.legend(handles=[acc_norm])
    plt.title('Norm of Accelerator', loc='left')

    if gt_info:
        plt.axvline(ts[start], linestyle='dashed', color='black')
        plt.axvline(ts[end], linestyle='dashed', color='black')
        plt.suptitle(pel + ' ' + ",  step counts(gt)={}".format(sc))

    plt.show()


def plot_1dim(ts, seq, Fs, dim, No, dtype='acc'):
    plt.subplot("33%d" % No)
    plt.plot(ts, seq)
    if dim == 2:
        plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (g)')
    plt.title(dtype + str(dim))
    # his
    plt.subplot("33%d" % (No + 1))
    plt.hist(seq, bins='auto')
    if dim == 2:
        plt.xlabel('Acceleration (g)')
    plt.ylabel('Count')
    plt.title(dtype + str(dim) + ' Distribution')

    # spectral
    plt.subplot("33%d" % (No + 2))
    NFFT = 2 ** nextpow2(seq.shape[0])
    Y = fft(seq, NFFT) / seq.shape[0]
    f = Fs / 2 * np.linspace(0, 1, int(NFFT / 2) + 1)
    plt.plot(f, 2 * np.abs(Y[:int(NFFT / 2) + 1]))
    plt.title(dtype + str(dim) + "Spectrum")
    if dim == 2:
        plt.xlabel("Frequency (Hz)")
    plt.ylabel("| Z(f) |")


def plot_amtn(data, gt_info, dtype='acc'):
    if dtype not in ['acc', 'gyro']:
        raise TypeError("Unsupported sensor type")

    ts, accs, gyros = data['sensor_readings']  # (ts, accs, gyros)}
    Fs = gt_info['Fs']
    for axis, No in enumerate(range(0, 9, 3)):
        plot_1dim(ts, accs[:, axis], Fs, axis, No + 1, dtype)
    plt.suptitle('subject{},age{},height{},weight{}, activity={}'.format(
        gt_info['subject'], gt_info['age'], gt_info['height'], gt_info['weight'], gt_info['activity']
    ))
    plt.show()


def nextpow2(n):
    l = int(np.log2(n))
    h = l + 1
    ld = np.abs(2 ** l - n)
    hd = np.abs(2 ** l - n)
    return l if ld < hd else h


# Test code of the previous functions
if __name__ == "__main__":
    # test Visual tools for Ubcomp13 data set
    from utils.ParseTraces import parseTrace
    import glob

    """
    ”*”匹配0个或多个字符；
    ”?”匹配单个字符；
    ”[ ]”匹配指定范围内的字符，如：[0-9]匹配数字
    """
    out = glob.glob(r"D:\WalkingTrajectoryEstimation\ubicomp13\*.out")
    for i in out:
        print(i)

    trace = out[50]

    data = parseTrace(trace)

    # plot_1trace(data)
    # test plot_1trace with gt_info
    from utils.datasets import WalkDetectDataset

    dataset = WalkDetectDataset(dir=r'D:\WalkingTrajectoryEstimation\ubicomp13')
    info, seqs = dataset[10]
    print(info)
    plot_1trace(seqs, info)

    print("--------Tools for Uicomp13 Ready--------")
    # test Visual tools for USC-HAD data set

    from utils.datasets import USCHAD

    had = USCHAD()

    info_dict, data_dict = had[100]
    plot_amtn(data_dict, info_dict)
    print("--------Tools for USC-HAD Ready--------")
