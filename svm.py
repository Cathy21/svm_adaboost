from libsvm.commonutil import svm_read_problem
import random
from statistics import mean
from libsvm.svmutil import svm_train, svm_predict


def make_k_folds(k, x, y):
    random.seed(1)
    folds = {}

    class0_idx = [i for i in range(len(y)) if y[i] == -1]
    class1_idx = [i for i in range(len(y)) if y[i] == 1]

    random.shuffle(class0_idx)
    random.shuffle(class1_idx)

    num_per_fold = int(len(y) / k / 2)

    for i in range(k):
        fold_x = []
        fold_y = []
        for j in range(num_per_fold):
            idx0 = class0_idx.pop()
            idx1 = class1_idx.pop()
            fold_x.append(x[idx0])
            fold_x.append(x[idx1])
            fold_y.append(y[idx0])
            fold_y.append(y[idx1])
        folds[i] = {'x': fold_x, 'y': fold_y}
    return folds

def kfold_cv(k, x, y):
    folds = make_k_folds(k, x, y)
    acc_lin = []
    acc_poly = []
    for i in range(k):
        val_x = folds[i]['x']
        val_y = folds[i]['y']
        train_x = []
        train_y = []
        for j in range(k):
            if j != i:
                train_x += folds[i]['x']
                train_y += folds[i]['y']
        # Linear
        m_lin = svm_train(train_y, train_x, '-t 0')
        (label, acc, val) = svm_predict(val_y, val_x, m_lin)
        acc_lin.append(acc[0])

        # Polynomial kernel
        m_poly = svm_train(train_y, train_x, '-t 1 -d 5')
        (label, acc, val) = svm_predict(val_y, val_x, m_poly)
        acc_poly.append(acc[0])

    return mean(acc_lin), mean(acc_poly)

def svm_all(x_train, y_train, x_test, y_test, opt):
    m = svm_train(y_train, x_train, opt)
    (label, train_acc, val) = svm_predict(y_train, x_train, m)
    (label, test_acc, val) =svm_predict(y_test, x_test, m)
    return train_acc[0], test_acc[0]


if __name__ == '__main__':
    y_train, x_train = svm_read_problem('./data/DogsVsCats.train')
    y_test, x_test = svm_read_problem('./data/DogsVsCats.test')
    acc_lin, acc_poly = kfold_cv(10, x_train, y_train)

    # Linear Kernel
    train_acc_lin, test_acc_lin = svm_all(x_train, y_train, x_test, y_test, '-t 0')

    # Polynomial Kernel
    train_acc_poly, test_acc_poly = svm_all(x_train, y_train, x_test, y_test, '-t 1 -d 5')
    print('10-fold CV Linear Kernel Accuracy: ' + str(acc_lin))
    print('10-fold CV Polynomial Kernel Accuracy: ' + str(acc_poly))
    print('Linear Kernel Training Accuracy: ' + str(train_acc_lin))
    print('Linear Kernel Testing Accuracy: '+str(test_acc_lin))
    print('Polynomial Kernel Training Accuracy: ' + str(train_acc_poly))
    print('Polynomial Kernel Testing Accuracy: ' + str(test_acc_poly))