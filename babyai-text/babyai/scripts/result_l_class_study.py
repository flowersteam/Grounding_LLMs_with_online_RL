import torch
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


def learning_curves(name_env, model_number):
    print("======== env:{} model:{}=======".format(name_env, model_number))
    log = pkl.load(open('storage/models/' + name_env + '/' + 'model_{}'.format(model_number) + '/log.pkl', "rb"))

    train_error = np.array(log["loss_cross_entropy_train"])
    success_rate_train = np.array(log["success_pred_train"])
    valid_error = np.array(log["loss_cross_entropy_valid"])
    success_rate_valid = np.array(log["success_pred_valid"])

    print('At epoch {} the CE error for train reach the minimum value of {}'.format(np.argmin(train_error),
                                                                                    min(train_error)))
    print(train_error)
    print(" ")
    print('At epoch {} the CE error for valid reach the minimum value of {}'.format(np.argmin(valid_error),
                                                                                    min(valid_error)))
    print(valid_error)
    print(" ")
    print('At epoch {} the success rate for train reach the maximum value of {}'.format(np.argmax(success_rate_train),
                                                                                        max(success_rate_train)))
    print(success_rate_train)
    print(" ")
    print('At epoch {} the success rate for valid reach the maximum value of {}'.format(np.argmax(success_rate_valid),
                                                                                        max(success_rate_valid)))
    print(success_rate_valid)

    """plt.plot(np.arange(len(train_error)), train_error)
    plt.title("Train error")
    plt.grid(axis='both')
    plt.show()
    plt.plot(np.arange(len(valid_error)), valid_error)
    plt.title("Valid error")
    plt.grid(axis='both')
    plt.show()
    plt.plot(np.arange(len(success_rate_train)), success_rate_train)
    plt.title("Success rate train set")
    plt.grid(axis='both')
    plt.show()
    plt.plot(np.arange(len(success_rate_valid)), success_rate_valid)
    plt.title("Success rate valid set")
    plt.grid(axis='both')
    plt.show()
"""




learning_curves('BabyAI-PutNextLocal-v0_no_answer_l_class', 0)

