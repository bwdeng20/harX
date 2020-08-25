import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optimzoo
import torch.nn.modules.loss as losszoo
import os
import numpy as np
from utils.evaluations import confusion_mat
import time


def find_newest(dire):
    files = os.listdir(dire)
    files.sort(key=lambda fn: os.path.getmtime(dire + "\\" + fn) if not os.path.isdir(dire + "\\" + fn) else 0)
    print("Coach will load the newest checkpoint file under the file dir \n \"{} \"".format(dire))
    return files[-1]


def str2optimizer(optimizer_name):
    """

    :param optimizer_name: the name of one optimizer
    :return: the optimizer class implemented in PyTorch
    """
    if hasattr(optimzoo, optimizer_name):
        return getattr(optimzoo, optimizer_name)
    else:
        raise Exception("Unsupported this optimizer: %s!" % optimizer_name)


def str2loss(loss_name):
    """

    :param loss_name: the name of one kind of loss
    :return:
    """
    if hasattr(losszoo, loss_name):
        return getattr(losszoo, loss_name)
    else:
        raise Exception("Unsupported Loss: %s !" % loss_name)


def xavier_init(layer):
    if type(layer) == nn.Conv2d:
        nn.init.xavier_normal_(layer.weight)


class CoachBase:
    def fit(self, _train_loader):
        raise NotImplementedError

    def resume(self, _train_loader, cp_file=None):
        raise NotImplementedError

    def evaluate(self, _test_loader):
        raise NotImplementedError


class Coach(CoachBase):

    def __init__(self, model, optimizer='Adam', loss='CrossEntropyLoss',
                 lr=0.001, epochs=10, metrics=None,
                 cpdir=r"./checkpoints/", logdir=r"./logs/", device=None):
        super(Coach, self).__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.last_epoch = 0
        self.last_loss = 0.0  # the loss of last epoch
        self.last_acc = 0.0
        self.best_acc = 0.0
        self.batches = 0  # the experienced mini-batches

        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.lr = lr

        if not os.path.exists(cpdir):
            os.mkdir(cpdir)
        self.cpdir = cpdir
        self.logdir = logdir + optimizer + '_lr=%5f_' % lr + loss
        self.writer = SummaryWriter(self.logdir)
        self.batches = 0

        self.evaluations = {'accuracy': None,
                            'precision': None,
                            'recall': None,
                            "confusionMatrix": None,
                            }
        if metrics:
            # more metrics stored in a dictionary will be unpacked and combine with the default 4 metrics(i.e,
            # accuracy, precision, recall, confusionMatrix )
            self.evaluations = {**self.evaluations, **metrics}
        pass

    def fit(self, _train_loader):
        self.model.train()

        batch_size = _train_loader.batch_size
        data_size = len(_train_loader.dataset)

        params = self.model.parameters()

        if isinstance(self.optimizer, str):
            optimizer = str2optimizer(self.optimizer)(params, self.lr)
        else:
            optimizer = self.optimizer

        if isinstance(self.loss, str):
            loss_layer = str2loss(self.loss)()
        else:
            loss_layer = self.loss

        loss_layer.to(self.device)
        for epoch in range(self.last_epoch + 1, self.epochs + 1):
            begin = time.time()
            epoch_loss = 0.0
            correct = 0.0
            for i, spl in enumerate(_train_loader):
                optimizer.zero_grad()
                inputs, labels, need_unpack = self.parse_sample(spl)
                if need_unpack:
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)

                predicted = torch.argmax(torch.softmax(outputs, dim=1), 1)
                correct += (torch.sum(labels == predicted)).item()

                # compute loss, backward, and update parameters
                loss = loss_layer(outputs, labels)
                loss.backward()
                optimizer.step()

                self.batches += 1
                batch_loss = loss.item() / batch_size
                epoch_loss += loss.item()
                if i % 19 == 0:  # print every 20 mini-batches
                    print('[epoch %5d, mini-batch %5d] batch_loss: %.5f' %
                          (epoch, i + 1, batch_loss))
                self.writer.add_scalar('Loss/batch', batch_loss, global_step=self.batches)

            # training information
            end = time.time()
            epoch_loss /= data_size
            epoch_acc = correct / data_size
            is_best = epoch_acc > self.best_acc
            if is_best:
                self.best_acc = epoch_acc
                self.model.save_weights(self.cpdir)  # actually the saved file is self.cpdir + model.name
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param.data.cpu().numpy(), epoch)

            self.last_acc = epoch_acc
            # save checkpoints if the accuracy improves
            self.model.save_checkpoint(epoch, self.batches, is_best,
                                       optimizer.state_dict(), epoch_loss, epoch_acc, self.cpdir)

            self.writer.add_scalar('Accuracy(train)', epoch_acc, global_step=epoch)
            self.writer.add_scalar('Loss/epoch', epoch_loss, global_step=epoch)
            print("Epoch %5d, AvgLoss: %5f ,Accuracy: %5f, Elapsed:  %5f" % (epoch, epoch_loss, epoch_acc, end - begin))

    def resume(self, _train_loader, cp_file=None):
        """

        :param _train_loader: a trainloader instance defined in torch.utils.data, detailed instructions about
                            it can be found at this Tutorial WebSite:
                            "https://pytorch.org/tutorials/beginner/data_loading_tutorial.html"

        :param cp_file: 1:a absolute file path

                        of one checkpoint file storing the weights, gradients, experienced
                        epochs, experienced mini batches, last epoch loss, last epoch accuracy, and the state of
                        optimizer!

                  or    2:a file directory

                        including many checkpoint files. In this case, the coach will
                        automatically find the most recently modified one.
        :return:
        """
        if os.path.isdir(cp_file):
            cp_file = cp_file + find_newest(cp_file)

        params = self.model.parameters()
        if isinstance(self.optimizer, str):
            self.optimizer = str2optimizer(self.optimizer)(params, self.lr)
        else:
            # this case assumes u have passed parameters into optimizer instance outside of this function
            pass

        self.last_epoch, self.batches, _, self.last_acc = self.model.load_checkpoint(self.optimizer, cp_file)
        self.fit(_train_loader)

    def evaluate(self, _test_loader):
        data_size = len(_test_loader.dataset)
        correct = 0
        total_loss = 0.0
        self.model.eval()
        if isinstance(self.loss, str):
            loss_layer = str2loss(self.loss)()
        else:
            loss_layer = self.loss
        loss_layer.to(self.device)
        LABEL = np.array([]).astype(np.int)
        PRED = np.array([]).astype(np.int)

        with torch.no_grad():
            for i, _sample in enumerate(_test_loader):
                inputs, labels, need_unpack = self.parse_sample(_sample)
                if need_unpack:
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                predicted = torch.argmax(torch.softmax(outputs, dim=1), 1)
                correct += (torch.sum(labels == predicted)).item()
                loss = loss_layer(outputs, labels)
                total_loss += loss.item()

                labels = labels.cpu().numpy().astype(np.int)
                predicted = predicted.cpu().numpy().astype(np.int)
                assert labels.shape == predicted.shape
                LABEL = np.append(LABEL, labels)
                PRED = np.append(PRED, predicted)

            print("Loss : %5f,Accuracy: %5f" % (total_loss / data_size, correct / data_size))

        confusion_matrix = confusion_mat(LABEL, PRED, self.model.C)

        self.evaluations['confusionMatrix'] = confusion_matrix
        self.evaluations['accuracy'] = correct / data_size

    def parse_sample(self, _sample):
        """
        in some circumstances, the input would be multi-path, meaning some components will be feed into the model
        at the beginning(the first entrance)  while u prefer to input the other components through other entrances
        (at sometimes). Since a tuple instance can not be "to" 'device' directly, we unpacked the components before
        transferring them into 'device'.
        Example:
                Ignatov CNN defined in .../harX/model/MoDeX.py/IgnatovCNN. This model concatenates some statistical
                information(e.g, Mean,Standard Deviation) and the flattened hidden vector, thus the 'input' would be
                a tuple instance which consists of two parts, the sensor reading vectors and the statistical feature
                vectors. However, a tuple can't be transferring to 'device' so we need some parsing strategies.

        :param _sample: a dictionary comprising 'input' and 'label'
        :return: the parsed input elements and labels transferring to the 'device'(either GPU or CPU)
        """
        need_unpack = 0
        if type(_sample['input']) == tuple or type(_sample['input']) == list:
            need_unpack = 1
            return tuple(elem.to(self.device) for elem in _sample['input']), _sample['label'].to(
                self.device) - 1, need_unpack
        else:
            return _sample['input'].to(self.device), _sample['label'].to(self.device) - 1, need_unpack


# if __name__ == "__main__":
#     # test for Coach.fit() passed!
#     from utils.datasets import UCIHAR
#     from utils.preprocess import Si2Ai
#     from torch.utils.data import DataLoader
#     from model.MoDeX import JY15CNN2
#
#     ucihar = UCIHAR(transform=Si2Ai(magnitude=True))
#     trainloader = DataLoader(ucihar, batch_size=30, shuffle=True, num_workers=2)
#     dataiter = iter(trainloader)
#     sample = next(dataiter)
#     print(sample['input'].shape)
#     print(sample['label'])
#
#     jy15cnn22 = JY15CNN2(6)
#     jy15cnn22.apply(xavier_init)
#     # NOTICE: to test resuming monitoring, please make sure that 'logdir' is the same as that when you new a Coach
#     # class to resume! A resume example can be found at
#     # Projects/integration_test/ResumeTest.py
#     Coach = Coach(jy15cnn22, epochs=100, logdir=r'../logs/test_resuming_monitoring-')
#     Coach.fit(trainloader)
