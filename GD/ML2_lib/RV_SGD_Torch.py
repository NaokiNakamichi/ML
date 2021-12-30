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

from tqdm import tqdm


class RVSGDByTorch:
    def __init__(self, lr, xtest=None, ytest=None, xvalid=None, yvalid=None):
        self.lr = lr
        self.model = nn.Module()
        self.x_test = xtest
        self.y_test = ytest
        self.x_valid = xvalid
        self.y_valid = yvalid

    def learn(self, k, x, y, model_type):
        class_num = int(max(y) + 1)
        n = x.shape[0]
        sep_num = n // k
        model_list = []
        w_dim = x.shape[1]
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
            # print(tmp_loss)
            valid_loss_store.append(valid.median_of_means(seq=np.array(tmp_loss), n_blocks=3))

        selected_index = np.argmin(valid_loss_store)

        return model_candidates[selected_index]

    def run_RVSGD(self, x_train, y_train, valid_x, valid_y, k, model_type):
        model_candidates = self.learn(k, x_train, y_train, model_type)
        result_model = self.valid(model_candidates=model_candidates, k=k, valid_x=valid_x, valid_y=valid_y)
        return result_model

    def prediction(self, x, y, model, is_print=False):
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


class RVSGDByTorchWithLoader:

    def __init__(self, lr, train_loader, test_loader, valid_dataset):
        self.lr = lr
        self.model = nn.Module()
        self.train_loeder = train_loader
        self.test_loeder = test_loader
        self.valid_dataset = valid_dataset

    def learn(self, k):
        n = len(self.train_loeder)
        sep_num = n // k
        model_list = []

        for i in tqdm(range(k)):
            model = models.CNN()
            model_candidate = SGDByTorch.SGDFromTrainLorder(lr=self.lr)
            loss_type = nn.CrossEntropyLoss()
            result_model = model_candidate.learn(train_loader=self.train_loeder, model=model, loss_type=loss_type,
                                                 )

            model_list.append(result_model)

        return model_list

    def valid(self, model_candidates, k):

        valid_loss_store = []
        sep_num = len(self.valid_dataset) // k
        loss_type = nn.CrossEntropyLoss()

        for j in range(k):
            tmp_loss = []
            val_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=sep_num, shuffle=True)
            for i in val_loader:
                data, target = i
                output = model_candidates[j](data)
                loss = loss_type(output, target)
                tmp_loss.append(loss.item())
            # print(tmp_loss)
            valid_loss_store.append(valid.median_of_means(seq=np.array(tmp_loss), n_blocks=3))

        selected_index = np.argmin(valid_loss_store)

        return model_candidates[selected_index]

    def run_RVSGD(self, k):
        model_candidates = self.learn(k)
        result_model = self.valid(model_candidates=model_candidates, k=k)
        return result_model

    def prediction(self, model):

        accuracy = SGDByTorch.SGDFromTrainLorder(lr=self.lr).test_accuracy(test_loader=self.test_loeder,model=model)
        return accuracy
