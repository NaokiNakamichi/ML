import numpy as np
from . import valid
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import copy
from sklearn.metrics import confusion_matrix, classification_report


class SGDTorch:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x, y, model, class_num,X_test = None,Y_test=None):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        feature_num = x.shape[1]
        w_stack = []
        sample_num = y.shape[0]
        model.parameter_init()
        swa_model = AveragedModel(model)
        loss_stack = []
        test_loss_stock = []
        accuracy_stack = []
        last = 0
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

            # 正解率の計算
            Y_test = torch.LongTensor(Y_test)
            if X_test is not None:
                with torch.no_grad():
                    y_test_pred = model(torch.tensor(X_test).float())
                    testloss = F.nll_loss(y_test_pred, Y_test)
                    prediction = y_test_pred.data.max(1)[1]

                    test_loss_stock.append(testloss.item())

                    if j % 10 == 0:

                        # print(f"prediction : {prediction.item()} y : {y_j}")
                        print(f"step : {j}")
                        accuracy = prediction.eq(Y_test).sum().numpy() / Y_test.shape[0]
                        print(confusion_matrix(Y_test, prediction))
                        print(classification_report(Y_test, prediction))
                        print(f"正解率 : {accuracy}")

            w_stack.append(swa_model)
            loss_stack.append(loss.item())
            last = model.state_dict()
            accuracy_stack.append(prediction.eq(Y_test).sum().numpy() / Y_test.shape[0])

        return model, w_stack, loss_stack, test_loss_stock, accuracy_stack

    def transition(self, k, train_x, train_y, transition_x, transition_y, model):
        _, w_stack, b_stack = self.learn(k, train_x, train_y, model)
        x = torch.tensor(transition_x).float()
        y = torch.LongTensor(transition_y)
        train_x = torch.tensor(train_x).float()
        train_y = torch.LongTensor(train_y)
        w_tmp = torch.stack(w_stack).permute(1, 0, 2, 3)
        b_tmp = torch.stack(b_stack).permute(1, 0, 2)
        half_num = train_x.shape[0] // 2
        loss_transition = []

        sep_num = w_tmp.shape[0]
        for update_i in range(2, w_tmp.shape[0]):
            valid_loss_store = []
            w_list = w_tmp[:update_i].mean(0)
            b_list = b_tmp[:update_i].mean(0)
            for i in range(k):
                tmp_loss = []
                for j in range(k):
                    with torch.no_grad():
                        model.fc1.weight.data = w_list[i]
                        model.fc1.bias.data = b_list[i]
                    output = model(train_x[j + half_num:j + half_num + sep_num])

                    loss = F.nll_loss(output, train_y[j + half_num:j + half_num + sep_num])
                    tmp_loss.append(loss.item())

                # print(tmp_loss)
                # print(w_list)
                valid_loss_store.append(valid.median_of_means(seq=np.array(tmp_loss), n_blocks=3))
            index = np.argmin(valid_loss_store)
            w_rv = w_list[index]
            b_rv = b_list[index]
            with torch.no_grad():
                model.fc1.weight.data = w_rv
                model.fc1.bias.data = b_rv

            tr_output = model(x)
            tr_loss = F.nll_loss(tr_output, y).item()

            loss_transition.append(tr_loss)

        return loss_transition, model

    def multiple_k_transition(self, k_list, train_x, train_y, transition_x, transition_y, model):
        k_transition = []
        for k in k_list:
            k_transition.append(self.transition(k=k, train_x=train_x, train_y=train_y, transition_x=transition_x,
                                                transition_y=transition_y,
                                                model=model)[0])

        return k_transition

    def multiple_k_accuracy(self, k_list, train_x, train_y, transition_x, transition_y, model):
        k_accuracy = []
        for k in k_list:
            m = self.transition(k=k, train_x=train_x, train_y=train_y, transition_x=transition_x,
                                transition_y=transition_y,
                                model=model)[1]

            x = torch.tensor(transition_x).float()
            y = torch.LongTensor(transition_y)
            output = m(x)
            prediction = output.data.max(1)[1]
            accuracy = prediction.eq(y).sum().numpy() / y.shape[0]
            k_accuracy.append(accuracy)

        return k_accuracy
