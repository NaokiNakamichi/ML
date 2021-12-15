from ML2_lib import SGDByTorch
from ML2_lib import models
from ML2_lib import format_data
from tqdm import tqdm
import torch
import torch.nn.functional as F


class ExManyTrials:
    def __init__(self, name_path, train_path, test_path):
        self.name_path = name_path
        self.train_path = train_path
        self.test_path = test_path

    def one_setting_ex(self, trial_num, lr, model):
        name_path = self.name_path
        train_path = self.train_path
        test_path = self.test_path

        accuracy_stack = []
        test_loss_stack = []

        for _ in tqdm(range(trial_num)):
            model.parameter_init()
            formatter = format_data.Format(train_path=train_path, name_path=name_path, test_path=test_path)
            X_train, y_train, X_test, y_test = formatter.data_return()
            class_num = int(max(y_train) + 1)
            learner = SGDByTorch.SGDAveTorch(lr=lr)
            result_model = learner.learn(x=X_train,
                                         y=y_train,
                                         model=model,
                                         class_num=class_num
                                         )

            y_test = torch.LongTensor(y_test)
            with torch.no_grad():
                y_test_pred = result_model(torch.tensor(X_test).float())
                testloss = F.nll_loss(y_test_pred, y_test)
                test_loss_stack.append(testloss.item())
                prediction = y_test_pred.data.max(1)[1]
                accuracy = prediction.eq(y_test).sum().numpy() / y_test.shape[0]
                accuracy_stack.append(accuracy)

        return accuracy_stack, test_loss_stack

    def ex_many_settings(self, trial_num, model_name_list, lr_list, model_list):

        col_name_list = []
        result_accuracy = []
        result_test_loss = []

        for lr in tqdm(lr_list):
            for model_name, model in zip(model_name_list, model_list):
                accuracy_stack, test_loss_stack = self.one_setting_ex(trial_num=trial_num, model=model, lr=lr)
                col_name_list.append(f"lr_{lr}_model_{model_name}")
                result_accuracy.append(accuracy_stack)
                result_test_loss.append(test_loss_stack)

        print(col_name_list)

        return result_accuracy, result_test_loss
