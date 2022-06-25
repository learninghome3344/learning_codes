import argparse
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

from model import FM
from ..common.utils import create_criteo_dataset, printbar


def train_model(model, ds_train, ds_valid, epoches):
    for epoch in tf.range(1,epoches+1):
        model.reset_metrics()

        # 在后期降低学习率
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr/2.0)
            tf.print("Lowering optimizer Learning Rate...\n\n")

        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y, reset_metrics=False)

        if epoch % 1 == 0:
            printbar()
            tf.print("epoch = ", epoch)
            print("train:", dict(zip(model.metrics_names, train_result)))
            print("vaild:", dict(zip(model.metrics_names, valid_result)))
            print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='命令行参数')
    parser.add_argument('-k', type=int, help='v_dim', default=8)
    parser.add_argument('-w_reg', type=float, help='w正则', default=1e-4)
    parser.add_argument('-v_reg', type=float, help='v正则', default=1e-4)
    args=parser.parse_args()

    file_path = '../data/train.txt'
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.5)

    k = args.k
    w_reg = args.w_reg
    v_reg = args.v_reg
    model = FM(k, w_reg, v_reg)
    model.summary()

    model.compile(optimizer=optimizers.Nadam(),
                loss=losses.SparseCategoricalCrossentropy(),
                metrics=[metrics.SparseCategoricalAccuracy(),metrics.SparseTopKCategoricalAccuracy(5)]) 
 

    #评估
    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("AUC: ", accuracy_score(y_test, pre))