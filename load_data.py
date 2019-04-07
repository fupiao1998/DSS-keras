from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import argparse


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtrain", "--data_path", required=True,
                    help="path to input image")
    ap.add_argument("-dlabel", "--label_path", required=True,
                    help="path to input label")
    ap.add_argument("-npath", "--npy_path", required=True,
                    help="path to .npy files")
    ap.add_argument("-itype", "--img_type", required=True,
                    help="path to output model")
    ap.add_argument("-r", "--rows", required=True, type=int,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", required=True, type=int,
                    help="shape of cols of input image")
    args = vars(ap.parse_args())
    return args


def create_train_data(data_path, img_type, rows, cols, label_path, npy_path):
    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    imgs = glob.glob(data_path + "//*." + img_type)
    imgdatas = np.ndarray((len(imgs), rows, cols, 3), dtype=np.uint8)
    imglabels = np.ndarray((len(imgs), rows, cols, 1), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        img = load_img(data_path + "\\" + midname)
        img = img_to_array(img)
        label = load_img(label_path + "\\" + midname, grayscale=True)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    np.save(npy_path + '\\X_train.npy', imgdatas)
    np.save(npy_path + '\\y_train.npy', imglabels)
    print('Saving to .npy files done.')


if __name__ == "__main__":
    args = args_parse()
    data_path = args["data_path"]
    label_path = args["label_path"]
    npy_path = args["npy_path"]
    img_type = args["img_type"]
    rows = args["rows"]
    cols = args["cols"]
    create_train_data(data_path, img_type, rows, cols, label_path, npy_path)