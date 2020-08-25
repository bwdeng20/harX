import torch
import torch.nn as nn
from model.MoDeX import MX
import torch.optim as optimzoo
from tqdm import tqdm
from model.Loss import SpKL,L2norm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
Accurate Step Length Estimation for Pedestrian Dead Reckoning Localization Using Stacked Autoencoders
https://ieeexplore.ieee.org/abstract/document/8488496
"""

class AutoEnoder(nn.Module):
    def __init__(self, d_in, hidden, acfun=torch.sigmoid):
        super(AutoEnoder, self).__init__()
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, d_in)
        self.activate = acfun

    def forward(self, x):
        x = self.l1(x)
        a2 = self.activate(x)
        x = self.l2(a2)
        return self.activate(x), a2


class SA(MX):
    def __init__(self, d_in, hidden, layers=2, acfun=torch.sigmoid):
        super(SA, self).__init__('SA-SLE')
        self.net = nn.ModuleList()
        self.sas = layers
        for i in range(layers):
            self.net.append(AutoEnoder(d_in, hidden, acfun))
        self.net.append(nn.Linear(d_in, 1, bias=False))

    def forward(self, x):
        for i in range(self.sas):
            x = self.net[i](x)[0]
        x = self.net[-1](x)
        return x

    def forward2layer(self, x, idx):
        a_hidden = 0
        for i in range(idx + 1):
            x, a_hidden = self.net[i](x)[0], self.net[i](x)[1]
        return x, a_hidden


def str2optimizer(optimizer_name):
    """

    :param optimizer_name: the name of one optimizer
    :return: the optimizer class implemented in Pytorch
    """
    if hasattr(optimzoo, optimizer_name):
        return getattr(optimzoo, optimizer_name)
    else:
        raise Exception("Unsupported this optimizer: %s!" % optimizer_name)


class SATrainer(object):
    def __init__(self, model, optimiz='SGD', lr=0.001, epoches=10):
        self.model = model.to(device)
        self.epoches = epoches
        self.lr = lr
        self.optimiz = optimiz

    def _fit1ae(self, idx, trainloader, epoch):
        # no grad for the past layers
        for param in self.model.net[:idx].paramters():
            param.requires_grad = False

            # only optimize the idx-th auto encoder
        params = self.model.net[idx].parameters()
        optimizer = str2optimizer(self.optimiz)(params, self.lr)
        re_cnstrc_err = 0
        for i, data in tqdm(enumerate(trainloader)):
            inputs, _ = data  # to train auto encoder only needs seq itself.
            optimizer.zero_grad()
            inputs = inputs.to(device)
            # forward
            re, a_h = self.model.forward2layer(inputs, idx)
            loss = SpKL(re, inputs, a_h)  # inputs are targets for AutoEncoder
            loss.backward()
            optimizer.step()
            re_cnstrc_err += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d AE,  %5d] loss: %.3f' %
                      (epoch + 1, idx, i + 1, re_cnstrc_err / 10))
                re_cnstrc_err = 0.0

    def fit(self, trainloader):

        for epoch in range(self.epoches):
            for idx in range(len(self.model.net) - 1):
                self._fit1ae(idx, trainloader, epoch)

            # use label to do regression
            # no grad for the past sa layers
            for param in self.model.net[:-1].parameters():
                param.requires_grad = False

            params = self.model.net[-1].parameters()
            optimizer = str2optimizer(self.optimiz)(params, self.lr)
            # get the reference to weight of the regression layer
            theta = list(self.model.net[-1].parameters())[0]

            regress_err = 0
            for i, data in tqdm(enumerate(trainloader)):
                inputs, labels = data  # to train auto encoder only needs seq itself.
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward through the whole model including stacked AutoEncoders and a regression layer
                Lengths = self.model(inputs)
                loss = L2norm(Lengths, labels, theta)  # inputs are targets for AutoEncoder
                loss.backward()
                optimizer.step()
                regress_err += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, regression layer,  %5d] loss: %.3f' %
                          (epoch + 1, i + 1, regress_err / 10))
                    regress_err = 0.0


if __name__ == "__main__":
    # test SA

    sa = SA(4, 2, 4)
    ones = torch.ones(3, 4)
    print(sa(ones))
    print(sa.forward2layer(ones, 1)[0])
    print("SA no problem!!!")
    # check the SATrainer
    print(sa)
