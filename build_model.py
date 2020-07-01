#Peichenhao 20200701
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Reshape, Activation
from keras.layers.normalization import BatchNormalization


def build_model(inp_shape, k_size=3):
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=inp_shape)
    conv1 = Convolution3D(padding='same', filters=32, kernel_size=k_size)(data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Convolution3D(padding='same', filters=32, kernel_size=k_size)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv4 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv6 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv6)

    conv7 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(pool3)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv8)

    conv9 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(pool4)#pool4
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    up1 = UpSampling3D(size=(2, 2, 2))(conv9)
    conv10 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(up1)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    merged1 = concatenate([conv11, conv8], axis=merge_axis)
    conv12 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(merged1)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)

    up2 = UpSampling3D(size=(2, 2, 2))(conv12)#conv12
    conv13 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(up2)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    merged2 = concatenate([conv14, conv6], axis=merge_axis)
    conv15 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(merged2)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)

    up3 = UpSampling3D(size=(2, 2, 2))(conv15)
    conv16 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(up3)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv17 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv16)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    merged3 = concatenate([conv17, conv4], axis=merge_axis)
    conv18 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(merged3)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)

    up4 = UpSampling3D(size=(2, 2, 2))(conv18)
    conv19 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(up4)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    conv20 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv19)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    merged4 = concatenate([conv20, conv2], axis=merge_axis)
    conv21 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(merged4)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)

    conv22 = Convolution3D(padding='same', filters=2, kernel_size=k_size)(conv21)
    output = Reshape([-1, 2])(conv22)
    output = Activation('softmax')(output)
    output = Reshape(inp_shape[:-1] + (2,))(output)

    model = Model(data, output)
    return model
