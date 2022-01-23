
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from loss_functions import dice_coef_loss, mse_cutout
from tensorflow.keras.optimizers import Adam
from paths_and_params.params import Params
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


params = Params()


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    drop1 = Dropout(params.drop_out)(s1)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    drop2 = Dropout(params.drop_out)(s2)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    drop3 = Dropout(params.drop_out)(s3)
    s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64)
    drop4 = Dropout(params.drop_out)(s4)


    """ Bridge """
    # pool4 = MaxPooling2D(pool_size=params.pool_size)(drop4)
    # b1 = BatchNormalization()(pool4)
    # b1 = Conv2D(params.feature_maps * 16, params.kernel, activation='relu', padding='same',
    #                kernel_initializer='he_normal')(b1)
    # b1 = Conv2D(params.feature_maps * 16, params.kernel, activation='relu', padding='same',
    #                kernel_initializer='he_normal')(b1)

    b1 = vgg16.get_layer("block5_conv3").output  ## (32 x 32)

    b1 = Dropout(params.drop_out)(b1)


    """ Decoder Segmentation """
    d1_seg = decoder_block(b1, drop4, params.feature_maps * 8)                     ## (64 x 64)
    d2_seg = decoder_block(d1_seg, drop3, params.feature_maps * 4)                     ## (128 x 128)
    d3_seg = decoder_block(d2_seg, drop2, params.feature_maps * 2)                     ## (256 x 256)
    d4_seg = decoder_block(d3_seg, drop1, params.feature_maps)                      ## (512 x 512)


    """ Decoder Reconstruction"""
    d1_re = decoder_block(b1, drop4, params.feature_maps * 8)                     ## (64 x 64)
    d2_re = decoder_block(d1_re, drop3, params.feature_maps * 4)                     ## (128 x 128)
    d3_re = decoder_block(d2_re, drop2, params.feature_maps * 2)                     ## (256 x 256)
    d4_re = decoder_block(d3_re, drop1, params.feature_maps)                      ## (512 x 512)


    """ Decoder Classification"""
    flat = Flatten()(b1)
    cl7 = Dense(params.feature_maps * 2, activation='elu')(flat)
    drop_cl1 = Dropout(params.drop_out / 2)(cl7)
    cl8 = Dense(params.feature_maps, activation='elu')(drop_cl1)
    drop_cl2 = Dropout(params.drop_out / 5)(cl8)


    """ Output """
    seg_output = Conv2D(1, 1, activation='sigmoid', name="seg_loss")(d4_seg)  # output of segmentation task
    re_output = Conv2D(1, 1, activation='linear', name="re_loss")(d4_re)
    class_output = Dense(1, activation='sigmoid', name="class_loss")(drop_cl2)


    model = Model(inputs, [seg_output, re_output, class_output])

    losses = {"seg_loss": dice_coef_loss, "re_loss": mse_cutout, "class_loss": 'binary_crossentropy'}

    lossWeights = {"seg_loss": params.loss_weights[0], "re_loss": params.loss_weights[1],
                   "class_loss": params.loss_weights[2]}

    model.compile(optimizer=Adam(lr=params.lr), loss=losses,
                  loss_weights=lossWeights, run_eagerly=True)
    model.run_eagerly = True
    model.summary()
    return model

