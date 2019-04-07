from model.DSSnet.DSSblock import vgg_net
from model.DSSnet.DSSblock import side_output
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Conv2DTranspose
from keras.layers import Conv2D
from keras.layers import Activation
from keras.models import Model


def dssnet(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 3))
    vgg_parameters = [[128, 3, 2], [128, 3, 2], [256, 5, 3], [256, 5, 3], [512, 5, 3]]
    vgg, side_list = vgg_net(vgg_parameters)(inputs)
    # side_list 是直接从vgg中提取出的tensor
    # 大小分别为[8,8,512],   [16,16,512],   [32,32,256],   [64,64,256],   [128,128,128],   [256,256,128]
    #         side_list[5]  side_list[4]   side_list[3]   side_list[2]    side_list[1]     side_list[0]
    side5 = side_output(512, (7, 7), [32, 16, 8, 4])(side_list[5])     # level 8
    # side5[0]=256    side5[1]=128    side5[2]=64    side5[3]=32
    side4 = side_output(512, (5, 5), [16, 8, 4, 2])(side_list[4])   # level 16
    # side4[0]=256    side4[1]=128    side4[2]=64    side4[3]=32
    side3 = side_output(256, (5, 5), [8, 4])(concatenate([side_list[3], side5[3], side4[3]]))  # level 32
    # side3[0]=256    side3[1]=128
    side2 = side_output(256, (5, 5), [4, 2])(concatenate([side_list[2], side5[2], side4[2]]))  # level 64
    # side2[0]=256    side2[1]=128
    side1 = side_output(128, (3, 3))(concatenate([side_list[1], side5[1], side4[1], side3[1], side2[1]]))  # level 128
    side1 = Conv2DTranspose(1, (4, 4), strides=2, padding='same', use_bias=False, activation=None)(side1)
    side0 = side_output(128, (3, 3))(concatenate([side_list[0], side5[0], side4[0], side3[0], side2[0]]))  # level 256

    fuse = concatenate([side5[0], side4[0], side3[0], side2[0], side1, side0])
    fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(fuse)
    # outputs
    o1    = Activation('sigmoid', name='o1')(side5[0])
    o2    = Activation('sigmoid', name='o2')(side4[0])
    o3    = Activation('sigmoid', name='o3')(side3[0])
    o4    = Activation('sigmoid', name='o4')(side2[0])
    o5    = Activation('sigmoid', name='o5')(side1)
    o6    = Activation('sigmoid', name='o6')(side0)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)
    dssnet = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, o6, ofuse])

    return dssnet


if __name__ == '__main__':
    model = dssnet(256, 256)
    model.summary()