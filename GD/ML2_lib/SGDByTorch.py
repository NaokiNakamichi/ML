import numpy as np
from . import valid
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import copy
from sklearn.metrics import confusion_matrix, classification_report

from tqdm import tqdm_notebook as tqdm


class SGDTorch:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x, y, model, class_num, X_test=None, Y_test=None, early_stopping=0):

        if type(x) != torch.Tensor:
            x = torch.from_numpy(x.astype(np.float32)).clone()
        # x = torch.tensor(x).float()
        # y = torch.LongTensor(y)
        sample_num = y.shape[0]
        # model.parameter_init()
        accuracy = 0
        for j in range(sample_num):
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            # print(x[j])
            output = model(x[j]).reshape(-1, class_num)
            # 　リサイズ　tensor(0) -> tensor([0])
            y_j = torch.unsqueeze(y[j], 0)
            loss = F.nll_loss(output, y_j)

            # Back Propagation
            loss.backward()
            optimizer.step()

            self.model = model


            # 正解率の計算
            if X_test is not None:
                Y_test = torch.LongTensor(Y_test)
                with torch.no_grad():
                    y_test_pred = model(torch.tensor(X_test).float())
                    testloss = F.nll_loss(y_test_pred, Y_test)
                    prediction = y_test_pred.data.max(1)[1]

                    # test_loss_stock.append(testloss.item())

                    accuracy = prediction.eq(Y_test).sum().numpy() / Y_test.shape[0]

                    if j % 1000 == 0:
                        # print(f"prediction : {prediction.item()} y : {y_j}")
                        print(f"step : {j}")

                        print(confusion_matrix(Y_test, prediction))
                        print(classification_report(Y_test, prediction))
                        print(f"正解率 : {accuracy}")
                        print(f"test loss : {testloss}")

            if early_stopping != 0:
                if early_stopping == j:
                    return model, accuracy

        self.model = model

        return model, accuracy


class SGDAveTorch:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x, y, model, class_num, X_test=None, Y_test=None):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        sample_num = y.shape[0]
        model.parameter_init()
        swa_model = AveragedModel(model)

        for j in range(sample_num):
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            # print(x[j])
            output = model(x[j]).reshape(-1, class_num)
            # 　リサイズ　tensor(0) -> tensor([0])
            y_j = torch.unsqueeze(y[j], 0)
            loss = F.nll_loss(output, y_j)

            # print(loss)

            # Back Propagation
            loss.backward()
            optimizer.step()

            if j > 5:
                swa_model.update_parameters(model)

        return swa_model


class SGDTorchCheck:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x, y, model, class_num, X_test=None, Y_test=None):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        sample_num = y.shape[0]
        self.model = model
        self.model.parameter_init()
        loss_stack = []
        test_loss_stock = []
        accuracy_stack = []
        swa_model = AveragedModel(self.model)
        for j in range(sample_num):
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            # print(x[j])
            output = self.model(x[j]).reshape(-1, class_num)
            # 　リサイズ　tensor(0) -> tensor([0])
            y_j = torch.unsqueeze(y[j], 0)
            loss = F.nll_loss(output, y_j)

            # print(loss)

            # Back Propagation
            loss.backward()
            optimizer.step()
            if j > 5:
                swa_model.update_parameters(self.model)

            # 正解率の計算

            if X_test is not None:
                Y_test = torch.LongTensor(Y_test)

                with torch.no_grad():

                    y_test_pred = self.model(torch.tensor(X_test).float())
                    testloss = F.nll_loss(y_test_pred, Y_test)
                    prediction = y_test_pred.data.max(1)[1]

                    test_loss_stock.append(testloss.item())

                    if j % 1000 == 0:
                        # print(f"prediction : {prediction.item()} y : {y_j}")
                        print(f"step : {j}")
                        print(self.model.state_dict())
                        accuracy = prediction.eq(Y_test).sum().numpy() / Y_test.shape[0]
                        print(confusion_matrix(Y_test, prediction))
                        print(classification_report(Y_test, prediction))
                        print(f"正解率 : {accuracy}")
                        print(f"test loss : {testloss}")

            loss_stack.append(loss.item())
            last = self.model.state_dict()
            accuracy_stack.append(prediction.eq(Y_test).sum().numpy() / Y_test.shape[0])

        return self.model, loss_stack, test_loss_stock, accuracy_stack


class SGDFromTrainLorder:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, train_loader, model, loss_type, early_stopping=0):
        for j in tqdm(train_loader):
            inputs, labels = j
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            # print(x[j])
            output = model(inputs)
            # 　リサイズ　tensor(0) -> tensor([0])
            loss = loss_type(output, labels)

            # Back Propagation
            loss.backward()
            optimizer.step()

            if early_stopping != 0:
                if early_stopping == j:
                    return model
        self.model = model

        return model

    def test_accuracy(self,test_loader,model):
        correct = 0
        total = 0
        # 勾配を記憶せず（学習せずに）に計算を行う
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        accuracy = correct/total

        return accuracy

class SGDImage:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x,y, model, loss_type, early_stopping=0):
        sample_num = x.shape[0]
        for j in tqdm(range(sample_num)):
            inputs, labels = x[j], y[j]
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            # print(x[j])

            inputs = torch.unsqueeze(inputs, 0)
            inputs = torch.unsqueeze(inputs, 0)
            output = model(inputs.float())
            # 　リサイズ　tensor(0) -> tensor([0])
            # output = torch.unsqueeze(output,0)
            labels = torch.unsqueeze(labels, 0)
            loss = loss_type(output, labels)

            # Back Propagation
            loss.backward()
            optimizer.step()

            if early_stopping != 0:
                if early_stopping == j:
                    return model
        self.model = model

        return model

    def test_accuracy(self,test_loader):
        correct = 0
        total = 0
        # 勾配を記憶せず（学習せずに）に計算を行う
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        accuracy = correct/total

        return accuracy