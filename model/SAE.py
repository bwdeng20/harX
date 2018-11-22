import torch
import torch.nn as nn
from model.MoDeX import MX
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def SpKL(output, target, a2, beta=1, rho=0.05):
    quadratic = torch.pow(output - target, 2)
    rho_hat = torch.mean(a2, dim=0)
    rho = rho * torch.ones(a2.shape[1])
    KLs = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    loss = torch.sum(quadratic) / 0.5 + beta * torch.sum(KLs)
    return loss


def L2norm(output, target, theta, lbda=0.05):
    NL = theta.shape[1]
    quadratic = torch.pow(output - target, 2)
    L2 = theta.mm(theta.t())
    J = torch.sum(quadratic) / (2 * NL) + lbda * L2 / 2
    return J


class SATrainer(object):
    def __init__(self, model, optimiz='SGD', lr=0.001, epoches=10):
        self.model = model.to(device)
        self.epoches = epoches
        self.lr = lr
        self.optimiz = optimiz

    def _reflect(self, type, params):
        if type is 'SGD':
            return optim.SGD(params, self.lr)
        if type is 'Adam':
            return optim.Adam(params, self.lr)

    def _fit1ae(self, idx, trainloader, epoch):
        # no grad for the past layers
        for param in self.model.net[:idx].paramters():
            param.requires_grad = False

            # only optimize the idx-th auto encoder
        params = self.model.net[idx].parameters()
        optimizer = self._reflect(self.optimiz, params)
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
            optimizer = self._reflect(self.optimiz, params)
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
