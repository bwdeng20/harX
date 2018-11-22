import numpy as np


class Si2Ai:
    """
    This class transforms a 'Raw signal matrix' into 'Signal Image' and
    then do 2-D Discrete Fourier Transform (DFT)on it.

    Refer to the paper
    'Human Activity Recognition Using Wearable Sensors by Deep Convolutional Neural Networks'
    for more Details

    Raw signal matrix :   a 2D array  which stores permutated raw signals with a shape of
                        Reading type Num(9 for UCI-HAR data set)   * window_size (50Hz mul 2.56s=128)

    Signal Image:         a (Nsis-1) * windows_size matrix

    DFT Image:            has same shape of Signal Image mentioned above

    """

    # TODO: we can build the concatenation rule according to the number of sensor readings so that there is no need
    # TODO: to generate the rule for each Raw signal matrix by invoking the "Raw2Sig" function
    def __init__(self, magnitude=True):
        self.magnitude = magnitude

    def __call__(self, sample):
        # 9*128  # scalr
        reading, label = sample["input"], sample["label"]
        if self.magnitude:
            dft = (np.absolute(np.fft.fft2(Raw2Sig(reading)))).astype(np.float32)
        else:
            dft = (np.fft.fft2(Raw2Sig(reading))).real.astype(np.float32)
        return {"input": dft[np.newaxis, :], "label": label}


def Raw2Sig(Raw):
    Ns = len(Raw)
    i = 1
    j = i + 1
    SI = Raw[i - 1].reshape(1, -1)
    SIS = '%d' % i
    Nsis = 1

    while i != j:
        if j > Ns:
            j = 1
        elif ('%d%d' % (i, j) not in SIS) and ('%d%d' % (j, i) not in SIS):
            SI = np.concatenate((SI, Raw[j - 1].reshape(1, -1)))
            SIS += '%d' % j
            Nsis += 1
            i = j
            j = i + 1
        else:
            j += 1

    return SI[:-1]


if __name__ == "__main__":
    from utils.datasets import UCIHAR
    import matplotlib.pyplot as plt

    ucihar = UCIHAR(transform=Si2Ai(False))

    samples = [ucihar[1], ucihar[2000], ucihar[3000], ucihar[7000]]
    for i in range(len(samples)):
        sample = samples[i]
        dftimg, label = sample['input'].squeeze(), sample['label']
        print(label)
        plt.subplot('22%d' % (i + 1))
        plt.imshow(dftimg.astype(np.uint8), cmap='gray')
        plt.title("Activity Image of %s" % ucihar.activity[label])
    plt.suptitle("DFT of several SignalImages")
    plt.show()
