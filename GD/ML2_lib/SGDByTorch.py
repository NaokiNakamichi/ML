import numpy as np
from . import valid
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import copy


class SGDTorch:
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, x, y, model):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        w_stack = []
        b_stack = []
        sample_num = y.shape[0]
        model.parameter_init()
        swa_model = AveragedModel(model)
        w_transition = []
        for j in range(sample_num):
            print(j)
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            output = model(x[j])
            loss = F.nll_loss(output, y[j])

            # Back Propagation
            loss.backward()
            optimizer.step()

            if j > 5:
                swa_model.update_parameters(model)

            # 正解率の計算
            prediction = output.data.max(1)[1]
            accuracy = prediction.eq(y).sum().numpy() / y.shape[0]

            w_transition.append(copy.deepcopy(model))

            w_stack.append(torch.stack(w_transition))
            # :TODO 線形のため一層のみのスタックになって。要改善

        return model, w_stack, b_stack

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
