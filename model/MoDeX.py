import torch
import torch.nn as nn
import time
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class MX(nn.Module):
    def __init__(self, name=''):
        super(MX, self).__init__()
        self.name = name

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename + self.name + "-trained")
        print("=> Parameters Saved !")

    def load_weights(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state)
        print("=> Parameters Loaded !")

    def save_checkpoint(self, epoch, batches, is_best, optimizer_state, loss, acc, cp_dir=r"./cps/"):
        """

        :param epoch:           int,the serial number of epoch
        :param batches:         int,experienced mini-batches
        :param is_best:         Bool,indicates whether the performance improves
        :param optimizer_state: optimizer.state()
        :param loss:            float
        :param acc:             float
        :param cp_dir:          str
        :return:                None
        """
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

    def predict(self, x):
        # for the most common classification tasks, u can overwrite this function to satisfy your project
        self.eval()
        outputs = self.forward(x)
        return torch.argmax(torch.softmax(outputs, dim=1), 1)


class JY15CNN(MX):
    """
    Model proposed in this paper
    -----------------------------------------------------------------------------------------
    Jiang, W., Yin, Z., 2015.
    Human activity recognition using wearable sensors by deep convolutional neural networks,
    in: MM, ACM. pp. 1307–1310.
    *****************************************************************************************
    Evaluation data set: UCI smart phone human activity recognition(Version1)
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    -----------------------------------------------------------------------------------------

    NOTICE: In the original paper, the input Activity Image is 36 * 68. But JY15CNN2,3
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
    N * 1 * 36 * 128    --5(out channels) * 5 * 5 -->
    N * 5 * 32 * 124    --4,4--->
    N * 5 * 8  *  31    --10(out channels) * 5 * 8-->
    N * 10 * 4 *  24    --2,4-->
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
    N * 1 * 36 * 128    --5(out channels) * 5 * 5 -->
    N * 5 * 32 * 124    --4,4--->
    N * 5 * 8  *  31    --10(out channels) * 5 * 8-->
    N * 10 * 4 *  24    --2,4-->
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
    -----------------------------------------------------------------------------------------
    [Ord´o˜nez and Roggen, 2016] Francisco Javier Ord´o˜nez and Daniel Roggen.
    Deep convolutional and lstm recurrent neural networks for multimodal wearable activity recognition.
    -----------------------------------------------------------------------------------------

    All Strides are 1 and All Padding are 0 by default here.
    input:  N *  1 * L(128 for UCIHar) * D(9 for UCIHar data set)
    conv1:  N * 64 * 124 * (D-1+2*0)/1+1=D
    conv2:  N * 64 * 120 * D
    conv3:  N * 64 * 116 * D
    conv4:  N * 64 * 112 * D
    permute:112 * N * (64*D)
    lstm1:  112 * N * 128(dropout=0.5)
    lstm2:  112 * N * 128(dropout=0.5)
    slice:  N * 128
    output: N * C
    """

    def __init__(self, C, D):
        """
        :param C: the number of activities to be recognised
        :param D: the sensor reading number
        """
        super(ConvLstm, self).__init__("ConvLstm")
        self.C = C
        self.D = D
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(64, 64, (5, 1))
        self.conv3 = nn.Conv2d(64, 64, (5, 1))
        self.conv4 = nn.Conv2d(64, 64, (5, 1))
        self.lstms = nn.LSTM(64 * D, 128, dropout=0.5, num_layers=2)
        self.out = nn.Linear(128, C)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.leaky_relu(x))
        x = self.conv3(F.leaky_relu(x))
        x = self.conv4(F.leaky_relu(x))
        x = x.permute(2, 0, 1, 3)
        x = x.reshape(112, -1, 64 * self.D)
        o, c = self.lstms(x)
        x = o[-1]  # slice
        return self.out(x)


class IgnatovCNN(MX):
    """
    Model proposed in this paper
    -----------------------------------------------------------------------------------------
    [Ignatov, 2018] Real-time human activity recognition from accelerometer data using
    Convolutional Neural Networks
    *****************************************************************************************
    Evaluation data set: UCI smart phone human activity recognition(Version1)
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    -----------------------------------------------------------------------------------------
    INPUT: Nx [S * D] wherein N is the batch size, S is 128 for UCIHAR dataset, and D is the
    number of time series collected by 3-axial accelerometer and 3-axial gyroscope, amounting to 6 dimension.



    All Strides are 1 and All Padding are 0 by default here.
    input: N x D(6) x 1 x S(usually128)
    conv1(relu): out_channel=196,filter=1 x 16, stride=1 x 1, padding=0 x 8("SAME" in Tensorflow version)
    state:N x 196 x 1 x (128-16+2*8)
    maxpool: 1 x 4
    state: N x 196 x 1 x  32
    flatten & concat with feature vectors :
    state: N x (196*32+40)
    FC1(ReLu): 1024
    state: N x 1024
    Dropout: rate=0.05
    state: N x 1024
    FC2: C(category number, 12 for HAPT data set)
    softmax:
    state: N x C
    """

    def __init__(self, C, D):
        super(IgnatovCNN, self).__init__('IgnatovCNN')
        self.C = C
        self.D = D
        self.conv = nn.Conv2d(in_channels=D, out_channels=196, kernel_size=(1, 16),
                              padding=(0, 8))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(196 * 32 + 40, 1024)
        self.drop = nn.Dropout(p=0.05, inplace=True)
        self.fc2 = nn.Linear(1024, C)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, fea):
        x = self.conv(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = torch.cat((x, fea), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class YoSle(MX):
    """
    My one-stage step model for step counting and stride length estimating.
    Require step beginning and end time point to be labelled.
    """

    # TODO
    def __init__(self, C):
        super(YoSle, self).__init__("YoSle")
        self.C = C
        pass


if __name__ == "__main__":
    jy15cnn = JY15CNN(100)
    # jy15.save_checkpoint(10, True, {1: 1, 2: 4}, 0.3)
    inp = torch.randn(4, 1, 36, 68)
    print(inp.shape)
    out = jy15cnn(inp)
    print(out.shape)
    print("=>JY15CNN Ready Ready!\n")

    inp = torch.randn(4, 1, 36, 128)
    jy15cnn2 = JY15CNN2(6)
    print(inp.shape)
    print(jy15cnn2(inp).shape)
    print("=>JY15CNN2 Ready Ready!\n")

    inp = torch.randn(3, 1, 128, 6)
    conv_lstm = ConvLstm(10, 6)
    print(inp.shape)
    print(conv_lstm(inp).shape)
    print("=>ConvLstm Ready Ready!\n")

    inp = torch.randn(2, 6, 1, 128)
    fea = torch.randn(2, 40)
    igna_cnn = IgnatovCNN(C=12, D=6)
    print(inp.shape)
    print(igna_cnn(inp, fea).shape)
    print("=> IgnatovCNN Ready Ready!\n")
