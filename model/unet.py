### Tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import*
from tensorflow.keras.layers import*
from tensorflow.keras.optimizers import*
from tensorflow.keras import losses
from tensorflow.keras import metrics

#### UNet model implemented in one function
def u_net_model(filter_size = 16, nr_filters=16, input_shape = 1024, act = 'relu'):
    #act = 'relu'
    inp = tf.keras.Input(shape = (input_shape, 1))
    ############################  Going Down/ Encoding
    ### First convolution layer
    conv1 = Conv1D(filters =nr_filters,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer ='he_normal')(inp)
    conv1 = Conv1D(filters =nr_filters,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv1)
    pool1  = MaxPooling1D(pool_size = 2,
                         strides = 2,
                         padding = 'same')(conv1)
    ### Second convolution layer
    conv2 = Conv1D(filters =nr_filters*2,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv1D(filters =nr_filters*2,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv2)
    pool2  = MaxPooling1D(pool_size = 2,
                         strides = 2,
                         padding = 'same')(conv2)
    ### third convolution layer
    conv3 = Conv1D(filters =nr_filters*4,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv1D(filters =nr_filters*4,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv3)
    pool3  = MaxPooling1D(pool_size = 2,
                         strides = 2,
                         padding = 'same')(conv3)
    ### fourth convolution layer
    conv4 = Conv1D(filters =nr_filters*8,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv1D(filters =nr_filters*8,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv4)
    drop4  = Dropout(0.5)(conv4)
    pool4  = MaxPooling1D(pool_size = 2,
                         strides = 2,
                         padding = 'same')(drop4)
    ### Fifth convolution layer
    conv5 = Conv1D(filters =nr_filters*16,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv1D(filters =nr_filters*16,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv5)
    drop5  = Dropout(0.5)(conv5)
    ############################### Going up/ decoding
    ### First Upsampling
    up6 = Conv1D(filters =nr_filters*8,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(drop5)
    sam6 = UpSampling1D(size =2)(up6)
    merg6 = concatenate([drop4, sam6], axis = 2)
    conv6 = Conv1D(filters =nr_filters*8,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(merg6)
    conv6 = Conv1D(filters =nr_filters*8,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv6)
    ### Second Upsampling
    up7 = Conv1D(filters =nr_filters*4,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv6)
    sam7 = UpSampling1D(size =2)(up7)
    merg7 = concatenate([conv3, sam7], axis = 2)
    conv7 = Conv1D(filters =nr_filters*4,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(merg7)
    conv7 = Conv1D(filters =nr_filters*4,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv7)
    ### Second Upsampling
    up8 = Conv1D(filters =nr_filters*2,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv7)
    sam8 = UpSampling1D(size =2)(up8)
    merg8 = concatenate([conv2, sam8], axis = 2)
    conv8 = Conv1D(filters =nr_filters*2,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(merg8)
    conv8 = Conv1D(filters =nr_filters*2,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv8)
    ### third Upsampling
    up9 = Conv1D(filters =nr_filters,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv8)
    sam9 = UpSampling1D(size =2)(up9)
    merg9 = concatenate([conv1, sam9], axis = 2)
    conv9 = Conv1D(filters =nr_filters,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(merg9)
    conv9 = Conv1D(filters =nr_filters,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv1D(filters =2,
                   kernel_size =(filter_size),
                   strides=1,
                   activation=act,
                   padding = 'same',
                  kernel_initializer = 'he_normal')(conv9)
    ### Lasst
    conv10  = Conv1D(filters = 1,
                    kernel_size = 1, activation= None)(conv9)
    #### combile model
    model = tf.keras.Model(inputs = inp, outputs = conv10)
    model.compile(optimizer = Adam(lr=1e-4), loss = 'mean_squared_error', metrics = ['mae'])
    return model
##### UNet model implemented based on Functions
def u_net_encoder(input_, filter_size=4,
                  nr_filters=16,
                  act_function='relu',
                  drop_value=0.5,
                  pool_value=2,
                  stride_value=2,
                  drop_out=True,
                  batch_n=False,
                  pool=True):
    conv1 = Conv1D(filters=nr_filters,
                   kernel_size=(filter_size),
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal')(input_)
    if batch_n:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act_function)(conv1)
    conv1 = Conv1D(filters=nr_filters,
                   kernel_size=(filter_size),
                   strides=1,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    if batch_n:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act_function)(conv1)
    if drop_out:
        conv1 = Dropout(drop_value)(conv1)
    if pool:
        pool1 = MaxPooling1D(pool_size=2,
                             strides=2,
                             padding='same')(conv1)
        return pool1, conv1
    else:
        return conv1, conv1


def u_net_decoder(input_,
                  concatenate_input,
                  filter_size=4,
                  nr_filters=16,
                  act_function='relu',
                  drop_value=0.5,
                  drop_out=True,
                  batch_n=False):
    dec_out = Conv1D(filters=nr_filters,
                     kernel_size=(filter_size),
                     strides=1,
                     padding='same',
                     kernel_initializer='he_normal')(input_)
    if batch_n:
        dec_out = BatchNormalization()(dec_out)
    dec_out = Activation(act_function)(dec_out)
    dec_out = UpSampling1D(size=2)(dec_out)
    dec_out = concatenate([concatenate_input, dec_out], axis=2)
    if drop_out:
        dec_out = Dropout(drop_value)(dec_out)
    dec_out = Conv1D(filters=nr_filters,
                     kernel_size=(filter_size),
                     strides=1,
                     padding='same',
                     kernel_initializer='he_normal')(dec_out)
    if batch_n:
        dec_out = BatchNormalization()(dec_out)
    dec_out = Activation(act_function)(dec_out)
    dec_out = Conv1D(filters=nr_filters,
                     kernel_size=(filter_size),
                     strides=1,
                     padding='same',
                     kernel_initializer='he_normal')(dec_out)
    if batch_n:
        dec_out = BatchNormalization()(dec_out)
    dec_out = Activation(act_function)(dec_out)
    return dec_out


def u_net_final(input_, filter_size=16,
                nr_filters=2,
                act_function='relu',
                drop_value=0.5,
                drop_out=False,
                batch_n=False):
    output_ = Conv1D(filters=nr_filters,
                     kernel_size=(filter_size),
                     strides=1,
                     padding='same',
                     kernel_initializer='he_normal')(input_)
    if batch_n:
        output_ = BatchNormalization()(output_)
    output_ = Activation(act_function)(output_)
    if drop_out:
        output_ = Dropout(drop_value)(output_)
    return output_


def u_net_model_function(input_size=1024,
                         filter_size=16,
                         nr_filters=16):
    input_ = tf.keras.Input(shape=(input_size, 1))
    act_function = 'relu'
    drop_out = True
    batch_ = True
    ############## Encoder ###################################
    enc_out_1, enc_con_1 = u_net_encoder(input_,
                                         filter_size=filter_size,
                                         nr_filters=nr_filters,
                                         act_function=act_function,
                                         drop_value=0.5,
                                         pool_value=2,
                                         stride_value=1,
                                         drop_out=False,
                                         batch_n=batch_)
    enc_out_2, enc_con_2 = u_net_encoder(enc_out_1,
                                         filter_size=filter_size,
                                         nr_filters=2 * nr_filters,
                                         act_function=act_function,
                                         drop_value=0.5,
                                         pool_value=2,
                                         stride_value=1,
                                         drop_out=False,
                                         batch_n=batch_)
    enc_out_3, enc_con_3 = u_net_encoder(enc_out_2,
                                         filter_size=filter_size,
                                         nr_filters=4 * nr_filters,
                                         act_function=act_function,
                                         drop_value=0.5,
                                         pool_value=2,
                                         stride_value=1,
                                         drop_out=False,
                                         batch_n=batch_)
    enc_out_4, enc_con_4 = u_net_encoder(enc_out_3,
                                         filter_size=filter_size,
                                         nr_filters=8 * nr_filters,
                                         act_function=act_function,
                                         drop_value=0.5,
                                         pool_value=2,
                                         stride_value=1,
                                         drop_out=drop_out,
                                         batch_n=batch_)
    enc_out_5, enc_con_5 = u_net_encoder(enc_out_4,
                                         filter_size=filter_size,
                                         nr_filters=16 * nr_filters,
                                         act_function=act_function,
                                         drop_value=0.5,
                                         pool_value=2,
                                         stride_value=1,
                                         drop_out=drop_out,
                                         batch_n=batch_,
                                         pool=False)
    #########################################################
    ############# Decoder ###################################
    dec_out_1 = u_net_decoder(enc_out_5,
                              enc_con_4,
                              filter_size=filter_size,
                              nr_filters=8 * nr_filters,
                              act_function=act_function,
                              drop_value=0.5,
                              drop_out=False,
                              batch_n=batch_
                              )
    dec_out_2 = u_net_decoder(dec_out_1,
                              enc_con_3,
                              filter_size=filter_size,
                              nr_filters=4 * nr_filters,
                              act_function=act_function,
                              drop_value=0.5,
                              drop_out=False,
                              batch_n=batch_
                              )
    dec_out_3 = u_net_decoder(dec_out_2,
                              enc_con_2,
                              filter_size=filter_size,
                              nr_filters=2 * nr_filters,
                              act_function=act_function,
                              drop_value=0.5,
                              drop_out=False,
                              batch_n=batch_
                              )
    dec_out_4 = u_net_decoder(dec_out_3,
                              enc_con_1,
                              filter_size=filter_size,
                              nr_filters=nr_filters,
                              act_function=act_function,
                              drop_value=0.5,
                              drop_out=False,
                              batch_n=batch_
                              )
    additional_ = u_net_final(dec_out_4,
                              nr_filters=2,
                              act_function=act_function,
                              drop_value=0.5,
                              drop_out=False,
                              batch_n=batch_)
    output_ = Conv1D(filters=1,
                     kernel_size=1,
                     activation=None)(additional_)
    #### combile model
    model = tf.keras.Model(inputs=input_, outputs=output_)
    model.compile(optimizer=SGD(lr=1e-2), loss='mean_squared_error', metrics=['mae'])
    return model
if '__name__'=='__main__':
    ###### This shows how to call the unet model in order to use it with the pre-trained models
    ########## Noise reduction (trained with training dataset A for noise reduction)
    model_noise = u_net_model(filter_size=5, nr_filters=32, input_shape=1024)
    #model_noise.load_weights('pretrained_noisereduction.h5')
    ########## Background rejection (trained with training dataset B for background rejection)
    model_background = u_net_model(filter_size=5, nr_filters=32, input_shape=1024)
    #model_background.load_weights('pretrained_backgroundrejection.h5')
    ########## model trained with training dataset C for background rejection in combination with SERDS
    model_serds      = u_net_model(filter_size=5, nr_filters=32, input_shape=800)
    #model_serds.load_weights('pretrained_serds_backgroundrejection.h5')


