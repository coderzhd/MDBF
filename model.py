# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: model.py
# @Software: PyCharm


from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout
from keras.layers import Flatten, Dense, Activation, BatchNormalization, CuDNNGRU, CuDNNLSTM
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.layers.wrappers import Bidirectional

import attention


def base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')
    y = Embedding(output_dim=128, input_dim=21, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(64, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)



    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Flatten()(merge)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x,y])

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def BiGRU_base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    # output = Dense(out_length, activation='sigmoid', name='output')(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

from keras.layers import *
import keras.backend as K
import tensorflow as tf

class MultiHeadAttention(Layer):
    """
    # Input
        three 3D tensor: Q, K, V
        each with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, input0 steps, head number * head size)`.
    Note: The layer has been tested with Keras 2.3.1 (Tensorflow 1.14.0 as backend)
    Example:
        S_inputs = Input(shape=(None,), dtype='int32')
        embeddings = Embedding(max_features, 128)(S_inputs)
        result_seq = MultiHeadAttention(8,16)([embeddings,embeddings,embeddings]) # self Attention
        result_vec = GlobalMaxPool1D()(result_seq)
        result_vec = Dropout(0.5)(result_vec)
        outputs = Dense(1, activation='sigmoid')(result_vec)
    """

    def __init__(self, heads, size_per_head, key_size=None, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head

    def get_config(self):
        config = super().get_config()
        config['heads'] = self.heads
        config['size_per_head'] = self.size_per_head
        config['key_size'] = self.key_size
        return config

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(units=self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(units=self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(units=self.out_dim, use_bias=False)

    def call(self, inputs):
        Q_seq, K_seq, V_seq = inputs

        Q_seq = self.q_dense(Q_seq)
        K_seq = self.k_dense(K_seq)
        V_seq = self.v_dense(V_seq)

        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.heads, self.key_size))
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.heads, self.key_size))
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.heads, self.size_per_head))

        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # Attention
        A = tf.einsum('bjhd,bkhd->bhjk', Q_seq, K_seq) / self.key_size ** 0.5
        A = K.softmax(A)

        O_seq = tf.einsum('bhjk,bkhd->bjhd', A, V_seq)
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.out_dim))
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class LayerNormalization(Layer):

    def __init__(
            self,
            center=True,
            scale=True,
            epsilon=None,
            **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config['center'] = self.center
        config['scale'] = self.scale
        config['epsilon'] = self.epsilon
        return config

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

    def call(self, inputs):
        if self.center:
            beta = self.beta
        if self.scale:
            gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class TransformerBlock(Layer):
    """
    # Input
        3D tensor: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, input steps, head number * head size)`.
    Note: The layer has been tested with Keras 2.3.1 (Tensorflow 1.14.0 as backend)
    Example:
        S_inputs = Input(shape=(None,), dtype='int32')
        embeddings = Embedding(max_features, 128)(S_inputs)
        result_seq = TransformerBlock(8,16,128)(embeddings)
        result_vec = GlobalMaxPool1D()(result_seq)
        result_vec = Dropout(0.5)(result_vec)
        outputs = Dense(1, activation='sigmoid')(result_vec)
    """

    def __init__(self, heads, size_per_head, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.ff_dim = ff_dim
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config['heads'] = self.heads
        config['size_per_head'] = self.size_per_head
        config['ff_dim'] = self.ff_dim
        config['rate'] = self.rate
        return config

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)
        assert input_shape[-1] == self.heads * self.size_per_head
        self.att = MultiHeadAttention(heads=self.heads, size_per_head=self.size_per_head)
        self.ffn = Sequential([
            Dense(self.ff_dim, activation="relu"),
            Dense(self.heads * self.size_per_head),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

    def call(self, inputs):
        attn_output = self.att([inputs, inputs, inputs])
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

def Transformer_base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=128, input_dim=21, input_length=length)(main_input)

    x = TransformerBlock(8, 16, 128)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def CNNTransformer(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(48, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    at = TransformerBlock(5, 20, 100)(x)
    ac = Convolution1D(32, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    a = Concatenate(axis=-1)([at, ac])#[517,132]
    # apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    bt = TransformerBlock(6, 22, 132)(a)
    bc = Convolution1D(48, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(a)
    b = Concatenate(axis=-1)([bt, bc])#[517,180]
    # bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    ct = TransformerBlock(9, 20, 180)(b)
    cc = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(b)
    merge = Concatenate(axis=-1)([ct, cc])  # [517,180]
    # cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)

    # merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    # O_seq = TransformerBlock(8, 24, 192)(merge)

    x = Flatten()(merge)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])

    # output = Dense(out_length, activation='sigmoid', name='output')(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)


    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def UpCNNBiGRU(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    # output = Dense(out_length, activation='sigmoid', name='output')(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def DiladCNNBiGRU(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')


    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    # output = Dense(out_length, activation='sigmoid', name='output')(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def MKey_DiladCNNBiGRU(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(48, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def MKey_DiladCNNBiGRU_two32(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model




# import os
# from keras.layers import Layer
#
#
class Attention(Layer):
    def __init__(self, step_dim: int = 517,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def MKey_UnetBiGRU(length, out_length, para):
    # 与之前的结构一模一样。没有用到Unet
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    # y = Attention(step_dim=166)(y)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(64, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])

    # output = Dense(out_length, activation='sigmoid', name='output')(x)

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input, fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

    # x = Attention(step_dim=517)(x)
    # attention_size = 64
    # attention_out = attention(x, attention_size, False)
    # attn_layer = AttentionLayer(name='attention_layer')
    # with tf.Session() as sess:
    #     merge1 = merge.eval()
    #     x1 = x.eval()
    # attn_out, attn_states = attn_layer([merge1, x1])
    #
    # # Concat attention input and decoder GRU output
    # decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([x, attn_out])


class Self_Attention(Layer):

    def __init__(self, output_dim:int = 300, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

def MKey_CNNBiGRUAttention(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    # y = Attention(step_dim=166)(y)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(64, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)


    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Self_Attention(100)(x)
    x = GlobalAveragePooling1D()(x)
    # x = Flatten()(x)
    # x = Attention(step_dim=517)(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])

    # output = Dense(out_length, activation='sigmoid', name='output')(x)

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input, fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

    # x = Attention(step_dim=517)(x)
    # attention_size = 64
    # attention_out = attention(x, attention_size, False)
    # attn_layer = AttentionLayer(name='attention_layer')
    # with tf.Session() as sess:
    #     merge1 = merge.eval()
    #     x1 = x.eval()
    # attn_out, attn_states = attn_layer([merge1, x1])
    #
    # # Concat attention input and decoder GRU output
    # decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([x, attn_out])


class MultiHeadSelfAttention(Layer):
    """ This uses Bahadanau attention """

    def __init__(self, num_heads=8, weights_dim=64,**kwargs):
        """ Constructor: Initializes parameters of the Attention layer """

        # Initialize base class:
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

        # Initialize parameters of the layer:
        self.num_heads = num_heads
        self.weights_dim = weights_dim

        if self.weights_dim % self.num_heads != 0:
            raise ValueError(
                f"Weights dimension = {weights_dim} should be divisible by number of heads = {num_heads} to ensure proper division into sub-matrices")

        # We use this to divide the Q,K,V matrices into num_heads submatrices, to compute multi-headed attention
        self.sub_matrix_dim = self.weights_dim // self.num_heads

        """
            Note that all K,Q,V matrices and their respective weight matrices are initialized and computed as a whole
            This ensures somewhat of a parallel processing/vectorization
            After computing K,Q,V, we split these into num_heads submatrices for computing the different attentions
        """

        # Weight matrices for computing query, key and value (Note that we haven't defined an activation function anywhere)
        # Important: In keras units contain the shape of the output
        self.W_q = Dense(units=weights_dim)
        self.W_k = Dense(units=weights_dim)
        self.W_v = Dense(units=weights_dim)

    def get_config(self):
        """ Required for saving/loading the model """
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "weights_dim": self.weights_dim
            # All args of __init__() must be included here
        })
        return config

    def build(self, input_shape):
        """ Initializes various weights dynamically based on input_shape """
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        # Weight matrix for combining the output from multiple heads:
        # Takes in input of shape (batch_size, seq_len, weights_dim) returns output of shape (batch_size, seq_len, input_dim)
        self.W_h = Dense(units=input_dim)

    def attention(self, query, key, value):
        """ The main logic """
        # Compute the raw score = QK^T
        score = tf.matmul(query, key, transpose_b=True)

        # Scale by dimension of K
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)  # == DIM_KEY
        scaled_score = score / tf.math.sqrt(dim_key)

        # Weights are the softmax of scaled scores
        weights = tf.nn.softmax(scaled_score, axis=-1)

        # The final output of the attention layer (weighted sum of hidden states)
        output = tf.matmul(weights, value)

        return output, weights

    def separate_heads(self, x, batch_size):
        """
            Splits the given x into num_heads submatrices and returns the result as a concatenation of these sub-matrices
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.sub_matrix_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """ All computations take place here """

        batch_size = tf.shape(inputs)[0]

        # Compute Q = W_q*X
        query = self.W_q(inputs)  # (batch_size, seq_len, weights_dim)

        # Compute K = W_k*X
        key = self.W_k(inputs)  # (batch_size, seq_len, weights_dim)

        # Compute V = W_v*X
        value = self.W_v(inputs)  # (batch_size, seq_len, weights_dim)

        # Split into n_heads submatrices
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, sub_matrix_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, sub_matrix_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, sub_matrix_dim)

        # Compute attention (contains weights and attentions for all heads):
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, sub_matrix_dim)

        # Concatenate all attentions from different heads (squeeze the last dimension):
        concat_attention = tf.reshape(attention,
                                      (batch_size, -1, self.weights_dim))  # (batch_size, seq_len, weights_dim)

        # Use a weighted average of the attentions from different heads:
        output = self.W_h(concat_attention)  # (batch_size, seq_len, input_dim)

        return output

    def compute_output_shape(self, input_shape):
        print(input_shape)
        """ Specifies the output shape of the custom layer, without this, the model doesn't work """
        return input_shape


def MKey_CNNBiGRUMultiAttention(length, out_length, para):
    # 只是将GRU改为了LSTM
    # 效果最好的模型(千万不要乱动代码！)
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


from keras.engine import Layer, InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def call(self, x):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.softmax(ait)
        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3

# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])

def MKey_CNNBiLSTMMAttention(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    merge1 = tf.expand_dims(merge,1)
    merge1 = cbam_module(merge1)
    merge1 = GlobalAveragePooling2D()(merge1)
    merge = Concatenate(axis=-1)([merge,merge1])


    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = AttentionWithContext()(x)

    # x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def DCNN_BiLSTM(length, out_length, para):
    # 效果最好的模型
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def Key_Only(length, out_length, para):
    # 只保留结构特征
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(y)

    model = Model(inputs=fu_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# 膨胀卷积的消融实验
def CNN_BiLSTM_Key(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# 膨胀卷积（全为2时）的消融实验
def DCNN222_BiLSTM_Key(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# 膨胀卷积（全为4时）的消融实验
def DCNN444_BiLSTM_Key(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# 膨胀卷积（全为8时）的消融实验
def DCNN888_BiLSTM_Key(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# BiLSTM的消融实验--无BiLSTM
def DCNN248_Key(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    # x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# BiLSTM的消融实验--GRU
def DCNN248_GRU_Key(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = CuDNNGRU(50, return_sequences=True)(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# BiLSTM的消融实验--LSTM
def DCNN248_LSTM_Key(length, out_length, para):
    # 只是将GRU改为了LSTM
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fu_input = Input(shape=(166,), dtype='int64', name='fu_input')

    y = Embedding(output_dim=128, input_dim=3, input_length=166)(fu_input)
    y = Convolution1D(16, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=2)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=4)(y)
    y = BatchNormalization()(y)
    y = Convolution1D(32, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value),dilation_rate=8)(y)#这里改成了32
    y = BatchNormalization()(y)
    # y = Convolution1D(128, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    # y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    # 先反卷积
    # x = Convolution1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(x))

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = CuDNNLSTM(50, return_sequences=True)(merge)

    x = Flatten()(x)


    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)
    x = Concatenate(axis=1)([x, y])


    # output = Dense(out_length, activation='sigmoid', name='output')(x)


    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input,fu_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

