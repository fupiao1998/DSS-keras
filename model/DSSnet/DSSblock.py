from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import MaxPooling2D
from keras.models import Model
from model.DSSnet.conv_block import Conv_bn_relu

factor_list = [4, 8, 16, 32]


def side_output(num_filters, kernel_size, factor_list=None):
    def layer(low_input):
        # low = Conv2D(num_filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same')(low_input)
        # low = Conv2D(num_filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same')(low)
        low_output = Conv2D(1, kernel_size=(1, 1), activation=None, padding='same')(low_input)
        if factor_list:
            up_sample = []
            for factor in factor_list:
                up_sample += [Conv2DTranspose(1, (2*factor, 2*factor),
                                              strides=factor, padding='same',
                                              use_bias=False, activation=None)(low_output)]
            return up_sample
        else:
            return low_output
    return layer


layer_list = [[512, 3], [512, 1], [256, 3], [256, 5]]


list=[[128, 3, 2], [128, 3, 2], [256, 5, 3], [256, 5, 3], [512, 5, 3]]


def vgg_net(layer_list):
    def layer(input_tensor):
        layers = []
        side_list = [0]*len(layer_list)
        for i, info in enumerate(layer_list):
            num_filters = info[0]
            kernel_size = info[1]
            repeat_times = info[2]
            side_list[i] = repeat_times
            for i in range(repeat_times):
                layers += [Conv_bn_relu(num_filters, kernel_size)]
            layers += [MaxPooling2D((2, 2), strides=(2, 2), padding='same')]
        x =input_tensor
        for i in range(len(layers)):
            x = layers[i](x)
        out_side = [side_list[0]]
        [out_side.append(out_side[-1] + side_list[i + 1]) for i in range(4)]
        vgg = Model(input_tensor, x)
        side_layers = []
        print('out_side', out_side)
        for i in out_side:
            side_layers += [vgg.get_layer('activation_' + str(i)).output]
        side_layers += [vgg.get_layer('max_pooling2d_5').output]
        return vgg, side_layers
    return layer


if __name__ == '__main__':
    # low_input = Input((128, 128, 3))
    # high_input = Input((512, 512, 3))
    # low_output, fuse = side_output(128, kernel_size=3, factor=4)(low_input)
    # model = Model([low_input], [low_output, fuse])
    # model.summary()
    low_input = Input((256, 256, 3))
    list = [[128, 3, 2], [128, 3, 2], [256, 5, 3], [256, 5, 3], [512, 5, 3]]
    vgg, side_list = vgg_net(list)(low_input)
    # vgg.summary()
    print(side_list)
    # factor_list = [4, 8, 16, 32]
    # list, out = side_output(64, (3, 3), factor_list)(low_input)
    # print(list)