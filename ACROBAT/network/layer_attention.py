import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
import numpy as np
from mxnet.base import numeric_types
from mxnet import symbol

class Reconstruction2D(nn.HybridBlock):
    def __init__(self, in_channels = 1, block_grad = False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.block_grad = block_grad

    def hybrid_forward(self, F, x, flow):
        if self.block_grad:
            flow = F.BlockGrad(flow)
        grid = F.GridGenerator(data = flow.flip(axis = 1), transform_type = "warp")
        return F.BilinearSampler(x, grid)

class Reconstruction2DSmooth(nn.HybridBlock):
    def __init__(self, in_channels = 1, block_grad = False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.block_grad = block_grad

    def hybrid_forward(self, F, x, flow):
        if self.block_grad:
            flow = F.BlockGrad(flow)
        grid = F.GridGenerator(data = flow.flip(axis = 1), transform_type = "warp").clip(-1, 1)
        return F.BilinearSampler(x, grid)

class DeformableConv2D(nn.HybridBlock):
    """ Deformable Convolution 2D

    Parameters
    ----------
    channels : int
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of n ints
        Specifies the dimensions of the convolution window.
    strides: int or tuple/list of n ints,
        Specifies the strides of the convolution.
    padding : int or tuple/list of n ints,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: int or tuple/list of n ints,
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str,
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    """
    def __init__(self, channels, kernel_size, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCHW', num_deformable_group=1, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros', 
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            if isinstance(kernel_size, numeric_types):
                kernel_size = (kernel_size,)*2
            if isinstance(strides, numeric_types):
                strides = (strides,)*len(kernel_size)
            if isinstance(padding, numeric_types):
                padding = (padding,)*len(kernel_size)
            if isinstance(dilation, numeric_types):
                dilation = (dilation,)*len(kernel_size)
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout, 
                'num_deformable_group' : num_deformable_group}

            wshapes = [
                (),
                (channels, in_channels) + kernel_size,
                (channels,) 
            ]
            self.weight = self.params.get('weight', shape=wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=wshapes[2],
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, offset, weight, bias=None):
        if bias is None:
            act = F.contrib.DeformableConvolution(x, offset, weight, name='fwd', **self._kwargs)
        else:
            act = F.contrib.DeformableConvolution(x, offset, weight, bias, name='fwd', **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def _alias(self):
        return 'deformable_conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)



class HAAM(nn.HybridBlock):
    def __init__(self, in_channels=3,channels=3,kernel_size1=3,kernel_size2 = 5, kernel_size3 = 3,stride = 1,padding1 = 1,padding2 = 2,padding3 = 3
                , dilation1 = 1, dilation2 = 1, dilation3 = 3,prefix=None):
        super().__init__(prefix=prefix)
        with self.name_scope():
            self._in_channels=in_channels
            self._channels = channels
            self._kernel_size1 = kernel_size1
            self._kernel_size2 = kernel_size2
            self._kernel_size3 = kernel_size3
            self._stride=stride
            self._padding1=padding1
            self._padding2=padding2
            self._padding3=padding3
            self._dilation1=dilation1
            self._dilation2=dilation2
            self._dilation3=dilation3
            self.GAP=nn.GlobalAvgPool2D()
            self.fc1=nn.Dense(units=self._channels,in_units=2*self._channels)
            self.fc2=nn.Dense(units=self._channels,in_units=self._channels)
            # self.bn=nn.BatchNorm()
            self.activate = nn.LeakyReLU(0.1)
            self.act_sig=nn.Activation('tanh')
            self._conv1=self.conv(False,self._in_channels,self._channels, self._kernel_size1, self._stride, self._padding1, self._dilation1)
            self._conv2=self.conv(False,self._in_channels,self._channels, self._kernel_size2, self._stride, self._padding2, self._dilation2)
            self._conv3=self.conv(False,self._in_channels,self._channels, self._kernel_size3, self._stride, self._padding3, self._dilation3)
            self._conv4=self.conv(True,2*self._channels,self._channels, 1,1, 0, 1)
            self._conv5=self.conv(True,self._channels,self._channels, 1,1, 0, 1)
            self._conv6=self.conv(True,2*self._channels,1,1,1,0, 1)#C=1
            self._conv7=self.conv(True,2*self._channels,self._channels, 1,1, 0, 1)
    def conv(self,use_in_channels=True,in_channels=0, channels=3, kernel_size = 3, stride = 1, padding = 1, dilation = 1,activation=True):
        net = nn.HybridSequential()
        with net.name_scope():
            if not use_in_channels:
                net.add(nn.Conv2D(channels, kernel_size, stride, padding, dilation))
            else:
                net.add(nn.Conv2D(channels, kernel_size, stride, padding, dilation,in_channels=in_channels))
            # net.add(nn.BatchNorm())
            if activation:
                net.add(self.activate)
        return net
    def Channelblock(self,F,data):
        conv1 = self._conv2(data)
        conv2 = self._conv3(data)
        data3=F.concat(conv1, conv2, dim=1)
        data3 = self.GAP(data3)
        data3 = self.fc1(data3)
        # data3 = self.bn(data3)
        data3 = self.activate(data3)
        data3 = self.fc2(data3)
        data3 = self.act_sig(data3)
        a = F.expand_dims(F.expand_dims(data3,axis=2),axis=3)
        a1 = 1-data3
        a1 = F.expand_dims(F.expand_dims(a1,axis=2),axis=3)
        y = F.broadcast_mul(conv1, a)
        y1 = F.broadcast_mul(conv2, a1)
        data_a_a1 = F.concat(y, y1, dim=1)
        conv3 = self._conv4(data_a_a1)
        # return F.broadcast_like(a,conv1)#conv3
        return conv3
    # spatial attentation
    def Spatialblock(self,F,data, channel_data):
        conv1 = self._conv1(data)
        conv2 = self._conv5(conv1)
        # batch2 = self.bn(conv2)
        # LeakyReLU2 = self.activate(batch2)
        LeakyReLU2 = self.activate(conv2)
        data3 = F.concat(channel_data, LeakyReLU2, dim=1)
        data3 = self.activate(data3)
        data3 = self._conv6(data3)
        data3 = self.act_sig(data3)
        a = F.repeat(data3, repeats=self._channels,axis=1)
        y = F.broadcast_mul(a, channel_data)
        a1 = 1-data3
        a1 = F.repeat(a1, repeats=self._channels,axis=1)
        y1 = F.broadcast_mul(a1, LeakyReLU2)
        data_a_a1 = F.concat(y, y1,dim=1)
        conv3 = self._conv7(data_a_a1)
        return conv3
    def hybrid_forward(self, F, data):
        channel_data = self.Channelblock(F,data)
        haam_data = self.Spatialblock(F,data,channel_data)
        return haam_data