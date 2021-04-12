from libsvm.commonutil import svm_read_problem
import random
from statistics import mean
from libsvm.svmutil import svm_train, svm_predict
from numpy import log, exp, sign


def get_models(x, y, K):
    m = len(y)
    w = [1/m] * m
    a = []
    model = []
    for k in range(K):
        print('Iteration: ' + str(k + 1))
        model.append(svm_train(y, x, '-t 0'))
        (label, train_acc, val) = svm_predict(y, x, model[k])
        error = sum([w[i]*(label[i] != y[i]) for i in range(m)])
        a.append(0.5*log((1-error)/error))
        weight = [w[i]*exp(-a[k]*y[i]*label[i]) for i in range(m)]
        w = [weight[i] / sum(weight) for i in range(m)]
    return model, a


def adaboost(x_train, y_train, x_test, y_test, K):
    models, a = get_models(x_train, y_train, K)
    model_outputs = [0]*K
    for i, m in enumerate(models):
        (labels, acc, val) = svm_predict(y_test, x_test, m)
        model_outputs[i] = [a[i]*labels[j] for j in range(len(y_test))]
    labels = [sign(sum([model_outputs[l][j] for j in range(K)])) for l in range(len(y_test))]
    acc = sum([y_test[j] == labels[j] for j in range(len(y_test))])/len(y_test)
    return labels, acc


if __name__ == '__main__':
    y_train, x_train = svm_read_problem('./data/DogsVsCats.train')
    y_test, x_test = svm_read_problem('./data/DogsVsCats.test')
    labels, acc = adaboost(x_train, y_train, x_test, y_test, 10)
    print(acc)

    labels, acc = adaboost(x_train, y_train, x_test, y_test, 20)
    print(acc)




