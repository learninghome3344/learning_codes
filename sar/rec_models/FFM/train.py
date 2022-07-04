import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

from model import FFM
from utils import create_criteo_dataset, printbar

BATCH_SIZE = 32

def train_model(model, ds_train, ds_valid, epoches):
    for epoch in tf.range(1, epoches+1):
        model.reset_metrics()

        # decrease learning_rate after a few epoches
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr/2.0)
            tf.print("Lowering optimizer Learning Rate...\n\n")

        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        # for x, y in ds_valid:
            # valid_result = model.test_on_batch(x, y, reset_metrics=False)
            # valid_result = model(x, training=False)

        if epoch % 1 == 0:
            printbar()
            tf.print("epoch = ", epoch)
            print("train:", dict(zip(model.metrics_names, train_result)))
            # print("vaild:", dict(zip(model.metrics_names, valid_result)))
            print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='commandline params')
    parser.add_argument('-k', type=int, help='v_dim', default=8)
    parser.add_argument('-w_reg', type=float, help='w_reg', default=1e-4)
    parser.add_argument('-v_reg', type=float, help='v_reg', default=1e-4)
    args=parser.parse_args()
    k = args.k
    w_reg = args.w_reg
    v_reg = args.v_reg

    file_path = '../data/train.txt'
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.5)
    
    model = FFM(feature_columns, k, w_reg, v_reg)

    '''
    train_on_batch
    '''
    # ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    #             .batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    # ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    #             .batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # model.compile(optimizer=tf.keras.optimizers.SGD(1e-2), # tf.keras.optimizers.RMSprop(1e-3),
    #             loss=tf.keras.losses.BinaryCrossentropy(),
    #             metrics=[tf.keras.metrics.AUC()])
    # train_model(model, ds_train, ds_test, 100)

    '''
    tape back propagation
    '''
    optimizer = optimizers.SGD(0.01)
    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    

    model.summary()
    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("AUC: ", accuracy_score(y_test, pre))