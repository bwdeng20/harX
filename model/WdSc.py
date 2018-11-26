import numpy as np
from utils.datasets import norm3d


class PeakDetector(object):
    """
    Naive peak detection
    """

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
            raise ValueError("please feed in acceleration sequence at least 2*K+1 long")
        size = acc_norm.shape[0]
        peaks = np.zeros(size).astype(np.int)
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

        acc_xyz = np.array(data[1]).astype(np.float32)
        acc_norm = np.sqrt(np.sum(acc_xyz ** 2, axis=1))
        print(acc_norm.dtype)
        return self._naive_peaks(acc_norm)


class RobustPD(PeakDetector):
    """
    Step counting and walking detection Model proposed in the  paper
    -----------------------------------------------------------------------------------------
    F. Gu, K. Khoshelham, J. Shang, F. Yu, and Z. Wei,
    “Robust and accurate smartphone-based step counting for indoor localization,
    ” IEEE Sensors J., vol. 17, no. 11, pp. 3453–3460, Jun. 2017
    -----------------------------------------------------------------------------------------


    Attributes
        self.state(a dict)
            keys                        |      values
                                        |
            peaks                       |

            false_peaks                 |

            ts                          |
            sim                         |
            Ci                          |
            Ti                          |
            acc                         |
            motion                      |

    """

    def __init__(self, K=15, Tmin=0.3, Tmax=1, sim_i=-5, M=2, N=4, sigma_var=0.7):
        super(RobustPD, self).__init__(K)
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.sim_i = -sim_i
        self.M = M
        self.N = N
        self.sigma_var = sigma_var
        self.state = {'peaks': None, 'false_peaks': None, 'ts': None,
                      'sim': None, 'Ci': None, 'Ti': None, 'acc': None,
                      'motion': None}

    def _exist_peaks(self):
        if self.state['peaks'] is None:
            raise ValueError("No naive peaks. \
                           Please compute naive peaks before advanced feature computation")

    def _exist_motions(self):
        if self.state['motion'] is None:
            raise ValueError("No motion. \
                                  Please recognize motions before computing similarities")

    def motions(self, s):
        """
         To compute 'sim_i' for each acc norm peak, we need motions corresponding to each peak.
         It's convenient to dispatch one peak and its following recordings into one motion, meaning
         the peak between two kinds of motions will be regarded as the forgoing one.
        :param s: a sequence comprising activity labels for each sample point with the same length
                 as our acc norm sequence.

                 lets say there are totally 2 kinds of activities(still and walking)marked by number 0-1,
                 and our sample sequence consists of 10 data points
                 s:             [0,     0,  0,  0,    0,   1,   1,   1,   1,    1]

                 acc_norm:      [3.4, 3.6, 3.3, 4.5, 4.2, 6.3, 6.7, 7.0 ,5.9, 6.5]
                 s and acc_norm has the same length

        NOTICE: u have to invoke this function manually after initialization
        :return:
        """
        self.state['motion'] = s

    def _periodicity(self):

        self._exist_peaks()

        ts = self.state['ts']
        bin_peaks = self.state['peaks']

        ts_peaks = ts[np.bool8(bin_peaks)]
        self.state['Ti'] = ts_peaks[1:] - ts_peaks[:-1]
        return self.state['Ti']

    def _similarity(self):
        self._exist_peaks()
        self._exist_motions()

        acc = self.state['acc']
        peak_bin = np.bool8(self.state['peaks'])
        motions = self.state['motion']

        peak_accs = acc[peak_bin]
        peak_motions = motions[peak_bin]

        raw_sims = peak_accs[2:] - peak_accs[:-2]
        raw_sims = -np.absolute(raw_sims)
        # check if m_i and m_(i+2) is walking state
        idx = 0
        while idx < len(peak_motions) - 2:
            if peak_motions[idx] != 0 or peak_motions[idx + 2] != 0:  # assume 0 marks walking state
                raw_sims[idx] = float('-inf')
            idx += 1

        self.state['sim'] = raw_sims
        return raw_sims

    def _continuity(self):
        # TODO: test this function!
        N = self.N
        M = self.M
        sigma_var = self.sigma_var
        acc = self.state['acc']
        peak_bin = np.bool8(self.state['peaks'])
        # assume totally P+1 peaks detected
        peak_idx = np.argwhere(peak_bin == 1)  # [[idx_0,],[idx_1,],...,[idx_P,]]
        peak_idx = peak_idx.reshape(peak_idx.size)  # [idx_0,idx_1,...,idx_P]

        # acc(i) represents a acc window comprising of [readings[idx_i],...], which is followed by readings[idx_(i+1)]

        # compute variance of each window
        vars = np.zeros(peak_idx.size)
        for j in range(len(peak_idx)):
            head = peak_idx[j]
            try:
                tail = peak_idx[j + 1]
            except IndexError:  # for the last peak and its window
                tail = None
            acc_window = acc[head, tail]
            vars[j] = np.var(acc_window)

        i = 0
        C = np.zeros(peak_idx.shape)
        while i < N:
            var_wins = vars[:i + 2]
            if len(var_wins) >= M:
                if np.sum(var_wins > sigma_var) >= M:
                    C[i] = 1
            i += 1
        while N <= i < len(peak_idx):
            try:
                var_wins = vars[i - N + 1:i + 2]
            except IndexError:
                var_wins = vars[i - N + 1:]
            if len(var_wins) >= M:
                if np.sum(var_wins > sigma_var) >= M:
                    C[i] = 1
            i += 1
        self.state['Ci'] = C
        return C
        # while i < N:
        #     # all peak indices ahead of the i-th peak plus the following 2 peak indices
        #     indices = peak_idx[:i + 3]
        #     vars = np.zeros(indices.size - 1)
        #     if len(indices) >= M:
        #         for j in range(len(indices) - 1):
        #             head = indices[j]
        #             tail = indices[j + 1]
        #             acc_window = acc[head, tail]
        #             vars[j] = np.var(acc_window)
        #     if np.sum(vars > sigma_var) > M:
        #         C[i] = 1
        #     i += 1
        # while N <= i < len(peak_idx) - 1:
        #     indices = peak_idx[i - N + 1:i + 3]
        #     vars = np.zeros(indices.size - 1)
        #     for j in range(len(indices) - 1):
        #         head = indices[j]
        #         tail = indices[j + 1]
        #         acc_window = acc[head, tail]
        #         vars[j] = np.var(acc_window)
        #     if np.sum(vars > sigma_var) > M:
        #         C[i] = 1
        #     i += 1

    def __call__(self, data):
        """

        :param data: a tuple or list consists of ts(ts), accData, ts, gyroData, ts, magnData.
        :return:
        """
        if self.state['peaks']:
            self.clear()

        ts, acc_xyz = np.array(data[0]), np.array(data[1])
        self.state['acc_xyz'] = acc_xyz
        acc_norm = np.sqrt(np.sum(acc_xyz ** 2, axis=1)).astype(np.float32)

        self.state['ts'] = ts
        self.state['acc'] = acc_norm

        self._naive_peaks(acc_norm)

        self._periodicity()
        self._similarity()
        self._continuity()

        pass


if __name__ == "__main__":
    # test naive peak detector
    st = PeakDetector(K=50)
    from utils.datasets import WalkDetectDataset

    dataset = WalkDetectDataset(dir=r'D:\WalkingTrajectoryEstimation\ubicomp13')
    info, seq = dataset[10]
    peaks = st(seq)
    print(type(peaks[0]))
    print(sum(peaks))

    st2 = RobustPD()
    # test _periodicity
    st2(seq)
