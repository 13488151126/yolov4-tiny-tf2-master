
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K
from tensorflow.keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D, Dense, Attention, Softmax

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z", "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航", "空"
         ]


def Moblienet(input):
   

    # 208,208,3 -> 104,104,32
    x = _conv_block(input, 32, strides=(2, 2))

    # 104,104,32 -> 104,104,64
    x = _depthwise_conv_block(x, 64, 1,
                              strides=(2, 2), block_id=0)
    x = _depthwise_conv_block(x, 128, 1, block_id=1)

    # 104,104,64 -> 52,52,128
    x = _depthwise_conv_block(x, 128, 1,
                              strides=(2, 2), block_id=2)
    # 52,52,128 -> 52,52,128
    x = _depthwise_conv_block(x, 128, 1, block_id=3)

    # 52,52,128 -> 26,26,256
    x = _depthwise_conv_block(x, 256, 1,
                              strides=(2, 2), block_id=4)

    # 26,26,256 -> 26,26,256
    x = _depthwise_conv_block(x, 256, 1, block_id=5)
    feat1 = x

    # 26,26,256 -> 13,13,512
    x = _depthwise_conv_block(x, 512, 1,
                              strides=(2, 2), block_id=6)
    # 13,13,512 -> 13,13,512
    x = _depthwise_conv_block(x, 512, 1, block_id=7)
    feat2 = x

    return feat1, feat2


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu1(x):
    return K.relu(x, max_value=1)


def relu6(x):
    return K.relu(x, max_value=6)
