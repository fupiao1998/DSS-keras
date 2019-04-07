import cv2
import numpy as np
import glob
import argparse
from PIL import Image

data_path = 'D:\\all-PythonCodes\\DSS-net\\MSRA-B\\image\\'
label_path = 'D:\\all-PythonCodes\\DSS-net\\MSRA-B\\annotation\\'
img_save = 'D:\\all-PythonCodes\\DSS-net\\MSRA-B\\img_resize\\'
label_save = 'D:\\all-PythonCodes\\DSS-net\\MSRA-B\\label_resize\\'
imgs = glob.glob(data_path + "//*." + 'png')
print(len(imgs))
for imgname in imgs:
    midname = imgname[imgname.rindex("\\") + 1:]
    img = cv2.imread(data_path + "\\" + midname)
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(img, (320, 480))
        img = Image.fromarray(img)
        img = np.asarray(img.rotate(90, expand=True))
    if img.shape != (320, 480, 3):
        img = cv2.resize(img, (480, 320))
    save_name = img_save + '\\' +midname
    cv2.imwrite(save_name, img)
    label = cv2.imread(label_path + "\\" + midname)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY)
    if label.shape[0] > label.shape[1]:
        label = cv2.resize(label, (320, 480))
        label = Image.fromarray(label)
        label = np.asarray(label.rotate(90, expand=True))
    if label.shape != (320, 480):
        label = cv2.resize(label, (480, 320))
    save_name_label = label_save + '\\' + midname
    cv2.imwrite(save_name_label, label)
