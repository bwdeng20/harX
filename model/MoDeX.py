import torch
import torch.nn as nn
import time
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class MX(nn.Module):
    def __init__(self, name=''):
        super(MX, self).__init__()
        self.name = name
        # # Since all models are created to train by default, it doesn't need
        # # this flag. But if u are about to do inference, this flag can control
        # # the model state(eval or train) after loading parameters. Without this flag,
        # # we can not set the whole model including some parameters(e.g, parameters for BN and Dropout)
        # # ready for inference.
        # self.inference = inference

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename + self.name + "-trained")
        print("=> Parameters Saved !")

    def load_weights(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state)
        print("=> Parameters Loaded !")

    def save_checkpoint(self, epoch, batches, is_best, optimizer_state, loss, acc, cp_dir=r"./cps/"):
        t = time.strftime("%Y-%m-%d,%H-%M-%S", time.localtime())
        if is_best:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer_state,
                        'loss': loss,
                        'batches': batches,
                        'acc': acc},
                       cp_dir + self.name + '-' + t + ',epoch%d,loss%f,acc%f' % (epoch, loss, acc))  # save checkpoint
            print("=> Saved a new best")
        else:
            print("=>  Accuracy did not improve")

    def load_checkpoint(self, optimizer, cp_file):
        checkpoint = torch.load(cp_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        batches = checkpoint['batches']
        acc = checkpoint['acc']
        print("=> Ready at (epoch%d,loss%f,acc%f)" % (epoch, loss, acc))
        return epoch, batches, loss, acc


class JY15CNN(MX):
    """
    Model proposed in this paper

    Jiang, W., Yin, Z., 2015.
    Human activity recognition using wearable sensors by deep convolutional neural networks,
    in: MM, ACM. pp. 1307–1310.

    NOTICE: In the original paper, the input Activity Image is 36 * 68. In contrast, JY15CNN2,3
            did not inherit this manner as UCI-HAR(SmartPhone) Data set consists of fixed
            windows of (50Hz * 2.56s=) 128 sample points, which means Activity Image generated
            for YJ15CNN2,3 is 36 * 128.
    """

    def __init__(self, Na):
        super(JY15CNN, self).__init__('JY15CNN')
        self.C = Na

        self.conv1 = nn.Conv2d(1, 5, 5)
        self.downsample1 = nn.AvgPool2d(4, stride=4)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.downsample2 = nn.AvgPool2d(2, stride=2)
        self.flat = Flatten()
        self.fc = nn.Linear(120, Na)

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsample1(torch.sigmoid(x))
        x = self.conv2(x)
        x = self.downsample2(torch.sigmoid(x))
        x = self.flat(x)
        x = self.fc(x)
        return x


class JY15CNN2(MX):
    """
    For a   C *  H *  W picture, and K * F * F filters, the output tensor will be
            K *  H'[(H-F+2P)/S+1] * W'[(W-F+2P)/S+1]

    All Strides are 1 and All Padding are 0 by default here.
    N * 1 * 36 * 128    --5 * 5 * 5 -->
    N * 5 * 32 * 124    --4,4--->
    N * 5 * 8  *  31     --10 * 5 * 8-->
    N * 10 * 4 *  24     --2,4-->
    N * 10 * 2 *  6     --flat-->
    N * 120             --fc--->
    N * Na(Na=6 kinds of activities for UCI-HAR data set)
    """

    def __init__(self, Na):
        super(JY15CNN2, self).__init__('JY15CNN2')

        self.C = Na

        self.conv1 = nn.Conv2d(1, 5, 5)
        self.downsample1 = nn.MaxPool2d(4, stride=4)
        self.conv2 = nn.Conv2d(5, 10, (5, 8))
        self.downsample2 = nn.MaxPool2d((2, 4), stride=(2, 4))
        self.flat = Flatten()
        self.fc = nn.Linear(120, Na)

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsample1(torch.sigmoid(x))
        x = self.conv2(x)
        x = self.downsample2(torch.sigmoid(x))
        x = self.flat(x)
        x = self.fc(x)
        return x


class JY15CNN3(MX):
    """
    For a   C *  H *  W picture, and K * F * F filters, the output tensor will be
            K *  H'[(H-F+2P)/S+1] * W'[(W-F+2P)/S+1]

    All Strides are 1 and All Padding are 0 by default here.
    N * 1 * 36 * 128    --5 * 5 * 5 -->
    N * 5 * 32 * 124    --4,4--->
    N * 5 * 8  *  31     --10 * 5 * 8-->
    N * 10 * 4 *  24     --2,4-->
    N * 10 * 2 *  6     --flat-->
    N * 120             --fc--->
    N * Na(Na=6 kinds of activities for UCI-HAR data set)
    """

    def __init__(self, Na):
        super(JY15CNN3, self).__init__('JY15CNN3')

        self.C = Na

        self.conv1 = nn.Conv2d(1, 5, 5)
        self.downsample1 = nn.MaxPool2d(4, stride=4)
        self.conv2 = nn.Conv2d(5, 10, (5, 8))
        self.downsample2 = nn.MaxPool2d((2, 4), stride=(2, 4))
        self.flat = Flatten()
        self.fc = nn.Linear(120, Na)

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsample1(F.leaky_relu(x))
        x = self.conv2(x)
        x = self.downsample2(F.leaky_relu(x))
        x = self.flat(x)
        x = self.fc(x)
        return x


class ConvLstm(MX):
    """
    Model proposed in this paper

    [Ord´o˜nez and Roggen, 2016] Francisco Javier Ord´o˜nez and Daniel Roggen.
    Deep convolutional and lstm recurrent neural networks for multimodal wearable activity recognition.
    """

    def __init__(self, C):
        super(ConvLstm, self).__init__("ConvLstm")
        pass


class YoSle(MX):
    """
    My one-stage step model for step counting and stride length estimating.
    Require step beginning and end time point to be labelled.
    """

    def __init__(self, C):
        super(YoSle, self).__init__("YoSle")
        pass


if __name__ == "__main__":
    jy15cnn = JY15CNN(100)
    # jy15.save_checkpoint(10, True, {1: 1, 2: 4}, 0.3)
    x = torch.randn(4, 1, 36, 68)
    print(x.shape)
    out = jy15cnn(x)
    print(out.shape)
    print(jy15cnn)
    print("=>JY15CNN Ready Ready!")

    x = torch.randn(4, 1, 36, 128)
    jy15cnn2 = JY15CNN2(6)
    print(x.shape)
    print(jy15cnn2(x).shape)
    print("=>JY15CNN2 Ready Ready!")
