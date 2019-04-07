import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from loss_functions import cross_entropy_balanced, pixel_error
import argparse

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", required=True,
                    help="path to npy. files to train")
    ap.add_argument("-mpath", "--model_path", required=True,
                    help="path to save the output model")
    ap.add_argument("-name","--model_name", required=True,
                    help="output of model name")
    ap.add_argument("-r", "--rows", required=True, type=int, default=320,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", required=True, type=int, default=480,
                    help="shape of cols of input image")
    args = vars(ap.parse_args())
    return args
'''
-npath
C:D:\\all-PythonCodes\\RCFs\\RCF-keras\\building_data\\111\\
-mpath
D:\\all-PythonCodes\\RCFs\\RCF-keras\\
-name
ep002-loss0.05415.h5
-r
320
-c
480
'''
'''
-npath
D:\\all-PythonCodes\\RCFs\\RCF-keras\\building_data\\111\\
-mpath
D:\\all-PythonCodes\\RCFs\\RCF-keras\\
-name
resnext_rcf_build.h5
-r
256
-c
256
'''
def test(args):
    X_test = np.load(args["npy_path"] + 'imgs_test.npy')
    model = load_model(args["model_path"] + args["model_name"],
                       custom_objects={'cross_entropy_balanced': cross_entropy_balanced, 'pixel_error': pixel_error})
    print("Begin")
    for i in range(10):
        y_pred = model.predict(X_test[i].reshape((-1, 256, 256, 3)))[-1]
        y_pred[y_pred > 0.4]=1
        y_pred[y_pred < 0.05] = 0
        y_pred = y_pred.reshape((256, 256))
        plt.figure(figsize=(25, 25))
        plt.subplot(1, 2, 1)
        plt.imshow(X_test[i], cmap='binary')
        plt.subplot(1, 2, 2)
        plt.imshow(y_pred, cmap='binary')
        name = str(i) + '.jpg'
        plt.savefig(name)


if __name__ == "__main__":
    args = args_parse()
    test(args)
'''
def test(args):
    X_train = np.load(args["npy_path"] + 'X_train.npy')
    X_test = np.load(args["npy_path"] + 'imgs_test.npy')
    # X_val = np.load(args["npy_path"] + 'X_val.npy')
    y_train = np.load(args["npy_path"] + 'y_train.npy')
    y_test = np.load(args["npy_path"] + 'y_test.npy')
    # y_val = np.load(args["npy_path"] + 'y_val.npy')
    model = load_model(args["model_path"] + args["model_name"],
                       custom_objects={'cross_entropy_balanced': cross_entropy_balanced, 'pixel_error': pixel_error})
    # test all images from test.npy
    print("Begin")
    for i in range(2, 3):
        y_pred = model.predict(X_train[i].reshape((-1, 320, 480, 3)))[-1]

        y_pred = y_pred.reshape((320, 480))
        plt.figure(figsize=(25, 16))
        plt.subplot(1, 3, 1)
        plt.imshow(X_train[i], cmap='binary')
        plt.subplot(1, 3, 2)
        plt.imshow(y_train[i].reshape((320, 480)), cmap='binary')
        plt.subplot(1, 3, 3)
        plt.imshow(y_pred, cmap='binary')
        plt.show()
        # name = str(i) + '.jpg'
        # plt.savefig(name)


if __name__ == "__main__":
    args = args_parse()
    test(args)
'''
