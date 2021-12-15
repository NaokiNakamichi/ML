import numpy as np
from . import valid
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import copy
from sklearn.metrics import confusion_matrix, classification_report


class GDTorch:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x, y, model, class_num,X_test = None,Y_test=None):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        sample_num = y.shape[0]
        model.parameter_init()
        for j in range(sample_num):
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            # print(x[j])
            output = model(x).reshape(-1, class_num)
            # 　リサイズ　tensor(0) -> tensor([0])
            loss = F.nll_loss(output, y)

            # print(loss)

            # Back Propagation
            loss.backward()
            optimizer.step()

            # 正解率の計算
            if X_test is not None:
                Y_test = torch.LongTensor(Y_test)
                with torch.no_grad():
                    y_test_pred = model(torch.tensor(X_test).float())
                    testloss = F.nll_loss(y_test_pred, Y_test)
                    prediction = y_test_pred.data.max(1)[1]

                    # test_loss_stock.append(testloss.item())

                    if j % 1000 == 0:

                        # print(f"prediction : {prediction.item()} y : {y_j}")
                        print(f"step : {j}")
                        accuracy = prediction.eq(Y_test).sum().numpy() / Y_test.shape[0]
                        print(confusion_matrix(Y_test, prediction))
                        print(classification_report(Y_test, prediction))
                        print(f"正解率 : {accuracy}")
                        print(model.state_dict())



        return model


class GDAveTorch:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x, y, model, class_num,X_test = None,Y_test=None):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        sample_num = y.shape[0]
        model.parameter_init()
        swa_model = AveragedModel(model)

        for j in range(sample_num):
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            # print(x[j])
            output = model(x).reshape(-1, class_num)
            # 　リサイズ　tensor(0) -> tensor([0])
            loss = F.nll_loss(output, y)

            # print(loss)

            # Back Propagation
            loss.backward()
            optimizer.step()

            if j > 5:
                swa_model.update_parameters(model)



        return swa_model