import numpy as np
from utils.datasets import norm3d


class PeakDetector(object):
    def __init__(self, K):
        self.K = K
        self.state = {'peaks': None}

    def clear(self):
        for key in self.state:
            self.state[key] = None
        print("delete the memory of last peak detection done")

    def get(self, key):
        if key not in self.state:
            raise KeyError("No {} attributes here")
        return self.state[key]

    def _naive_peaks(self, acc_norm):
        if acc_norm.shape[0] < 2 * self.K + 1:
            raise ValueError("please feed acceleration sequence at least 2*K+1 long")
        size = acc_norm.shape[0]
        peaks = np.zeros(size)
        K = self.K
        idx = 0
        while (idx < K + 1):
            if np.all(acc_norm[idx] > acc_norm[idx + 1:idx + K + 1]):
                peaks[idx] = 1
                idx += K
            idx += 1

        while (K + 1 <= idx < size - K):
            post_peak = np.all(acc_norm[idx] > acc_norm[idx + 1:idx + K + 1])
            pre_peak = np.all(acc_norm[idx] > acc_norm[idx - K:idx])
            if pre_peak and post_peak:
                peaks[idx] = 1
                idx += K
            idx += 1
        while (size - K <= idx < size):
            if np.all(acc_norm[idx] > acc_norm[idx - K:idx]):
                peaks[idx] = 1
                idx += K
            idx += 1
        self.state['peaks'] = peaks
        return peaks

    def __call__(self, data):
        """

        :param data: a tuple or list consists of ts(ts), accData, ts, gyroData, ts, magnData.
        :return: indices wherein index is 1 if it corresponds to a peak, 0 otherwise.
        """
        if self.state['peaks']:
            self.clear()

        acc_xyz = np.array(data[1])
        acc_norm = np.sqrt(np.sum(acc_xyz ** 2, axis=1))

        return self._naive_peaks(acc_norm)


class RobustPD(PeakDetector):
    def __init__(self, K=15, Tmin=0.3, Tmax=1, sim_i=-5, M=2, N=4, sigma_var=0.7):
        super(PeakDetector, self).__init__(K)
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.sim_i = -sim_i
        self.M = M
        self.N = N
        self.sigma_var = sigma_var
        self.state = {'peaks': None, 'false_peaks': None, 'ts': None, \
                      'sim': None, 'Ci': None, 'Ti': None, 'acc': None,
                      'motion': None}

    def _exist_peaks(self):
        if not self.state['peaks']:
            raise ValueError("No naive peaks. \
                           Please compute naive peaks before advanced feature computation")

    def _periodicity(self):
        self._exist_peaks()

        ts = self.state['ts']
        idx_peaks = self.state['peaks']

        ts_peaks = ts[np.bool8(idx_peaks)]
        self.state['Ti'] = ts_peaks[1:] - ts_peaks[:-1]
        return self.state['Ti']


    def _similarity(self):
        self._exist_peaks()

        acc = self.state['acc']
        idx_peaks = self.state['peaks']

        self.state['sim'] = -np.abs()
        pass

    def __call__(self, data):
        #  method arguments should match with parent class method you're overriding.
        if self.state['peaks']:
            self.clear()

        ts, acc_xyz = data[0], data[1]
        acc_norm = np.sqrt(np.sum(acc_xyz ** 2, axis=1))

        self.state['ts'] = ts
        self.state['acc'] = acc_norm

        self._naive_peaks(acc_norm)
        self._periodicity()

        pass


if __name__ == "__main__":
    st = PeakDetector(K=50)
    from utils.datasets import WalkDetectDataset

    dataset = WalkDetectDataset(dir=r'D:\WalkingTrajectoryEstimation\ubicomp13')
    info, seq = dataset[10]
    peaks = st(seq)
    print(type(peaks[0]))
    print(sum(peaks))

    acc_norm = norm3d(seq[1])
