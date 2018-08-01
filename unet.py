from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization, Conv2DTranspose, add, LeakyReLU, Activation

'''
original Source code: https://github.com/pietz/unet-keras/blob/master/unet.py
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''

def conv_block(m, dim, acti, bn, res, do=0):

	shortcut=m
	n = Conv2D(dim, 3, activation='linear', padding='same')(m)
	n = BatchNormalization()(n) if bn else n
	n = LeakyReLU()(n) if acti=='LeakyReLU' else Activation(acti)(n)

	n = Conv2D(dim, 3, activation='linear', padding='same')(n)
	n = BatchNormalization()(n) if bn else n

	if res:
		shortcut= Conv2D(dim, kernel_size=(1, 1),  padding='same')(shortcut)
		shortcut = BatchNormalization()(shortcut) if bn else shortcut
		n=add([n, shortcut])

	n = LeakyReLU()(n) if acti == 'LeakyReLU' else Activation(acti)(n)
	n = Dropout(do)(n) if do else n
	return n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation='linear', padding='same')(m)
			m = LeakyReLU()(m) if acti == 'LeakyReLU' else Activation(acti)(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation='linear', padding='same')(m)
			m = LeakyReLU()(m) if acti == 'LeakyReLU' else Activation(acti)(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='LeakyReLU',
		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)
