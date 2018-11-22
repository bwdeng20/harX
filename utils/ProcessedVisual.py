import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def visual_pk(ts, acc_norm, gt_info, peaks, axes=None):
    """

    :param data: time stamps, acc_norm data
    :param gt_info: gt_info('p1.1Hand_held', [1607, 5393, 70])
    :param peaks: e.g, peaks=np.array([1,0,0,...,1]) wherein the i-th time stamp corresponds
                    to a peak(step) selected by our step counting algorithm.

    :param axes: the axes instance of matplotlib library
    :return:
    """

    # parse ground truth info
    pel, (start, end, gt_sc) = gt_info

    # get axes to plot something
    show_flag = False
    if not axes:
        show_flag = True
        axes = plt.gca()
        # if u call this func to plot a subplot(axes), the title should be conducted by an outer func.
        # Namely no need to execute the next statement.
        axes.set_title("{} , step counts(gt)={}, estimate={}".format(pel, gt_sc, sum(peaks)))

    # handle data
    if len(ts) != len(acc_norm):
        raise Exception("The ts sequence does not match the acc_norm sequence")
    ts = np.array(ts)
    acc_norm = np.array(acc_norm)
    # scatter data
    bools = np.bool8(peaks)
    ts_p = ts[bools]
    acc_p = acc_norm[bools]

    axes.plot(ts, acc_norm, label='acc_norm')
    axes.scatter(ts_p, acc_p, c='', alpha=0.8, marker='o', edgecolors='r')
    axes.axvline(ts[start], linestyle='dashed', color='black')
    axes.axvline(ts[end], linestyle='dashed', color='black')
    if show_flag:
        plt.show()


def visual_cm(mat, classes=None, axes=None,color_bar=None):
    show_flag = False
    if not axes:
        show_flag = True
        axes = plt.gca()

    H,W=mat.shape
    mat_ax=axes.imshow(mat,cmap=cm.coolwarm)
    plt.xticks(range(H),classes,rotation=90)
    plt.yticks(range(W),classes)
    axes.xaxis.set_ticks_position('top')
    if color_bar:
        plt.colorbar(mat_ax)

    if len(classes)<=40:

        for x in range(W):
            for y in range(H):
                axes.annotate(str(mat[x][y]),xy=(y,x),
                              horizontalalignment='center',
                              verticalalignment='center')
    else:
        print("Since too many categories we omit the number annotation")

    if show_flag:
        plt.show()


if __name__ == "__main__":
    from model.WdSc import PeakDetector
    from utils.datasets import WalkDetectDataset, norm3d

    st = PeakDetector(K=30)

    dataset = WalkDetectDataset(dir=r'D:\WalkingTrajectoryEstimation\ubicomp13')
    info, seq = dataset[30]
    peaks = st(seq)

    ts = seq[0]
    acc_norm = norm3d(seq[1])
    # test pass an axes instance into collect axes(subplots)
    axes = plt.gca()
    visual_pk(ts, acc_norm, info, peaks, axes)
    plt.show()

    # test no passing axes instance to\\\
    visual_pk(ts, acc_norm, info, peaks)
