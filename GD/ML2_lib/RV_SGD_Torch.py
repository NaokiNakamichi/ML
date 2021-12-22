import numpy as np

from . import valid

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from ML2_lib import SGDByTorch
from sklearn.metrics import confusion_matrix, classification_report

from ML2_lib import models

from ML2_lib import valid


class RVSGDByTorch:
    def __init__(self, lr, xtest=None, ytest=None):
        self.lr = lr
        self.model = nn.Module()
        self.x_test = xtest
        self.y_test = ytest

    def learn(self, k, x, y, model_type):
        class_num = int(max(y) + 1)
        n = x.shape[0]
        sep_num = n // k
        model_list = []
        w_dim = x.shape[1]
        class_num = 10
        unit_num = 512

        for i in range(k):
            if "linear" == model_type:
                model = models.LinearClassification(w_num=w_dim, c_num=class_num)

            elif "FF_L1" == model_type:
                model = models.FF_L1(w_num=w_dim, c_num=class_num, unit_num=unit_num)

            elif "FF_L2" == model_type:
                model = models.FF_L2(w_num=w_dim, c_num=class_num, unit_num=unit_num)

            else:
                raise ValueError("第二変数を確認、linear or FF_L1 or FF_L2")

            model.parameter_init()
            model_candidate = SGDByTorch.SGDTorch(lr=self.lr)
            result_model, _ = model_candidate.learn(x=x[i * sep_num:(i + 1) * sep_num],
                                                    y=y[i * sep_num:(i + 1) * sep_num], model=model,
                                                    class_num=class_num, X_test=self.x_test, Y_test=self.y_test)

            model_list.append(result_model)

        return model_list

    def valid(self, model_candidates, k, valid_x, valid_y):

        valid_num = valid_x.shape[0]
        sep_num = valid_num // k
        valid_loss_store = []

        for i in range(k):
            tmp_loss = []
            for j in range(k):
                with torch.no_grad():
                    output = model_candidates[i](valid_x[j * sep_num:j * sep_num + sep_num])
                    loss = F.nll_loss(output, valid_y[j * sep_num:j * sep_num + sep_num])
                    tmp_loss.append(loss.item())
            print(tmp_loss)
            valid_loss_store.append(valid.median_of_means(seq=np.array(tmp_loss), n_blocks=3))

        selected_index = np.argmin(valid_loss_store)

        return model_candidates[selected_index]

    def prediction(self, x, y, model,is_print=False):
        y = torch.LongTensor(y)
        with torch.no_grad():
            y_test_pred = model(torch.tensor(x).float())
            testloss = F.nll_loss(y_test_pred, y)
            prediction = y_test_pred.data.max(1)[1]

            # test_loss_stock.append(testloss.item())

            accuracy = prediction.eq(y).sum().numpy() / y.shape[0]

            # print(f"prediction : {prediction.item()} y : {y_j}"

            if is_print:
                print(confusion_matrix(y, prediction))
                print(classification_report(y, prediction))
                print(f"正解率 : {accuracy}")
                print(f"test loss : {testloss}")

        return testloss, accuracy
