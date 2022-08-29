import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from utils import *
import numpy as np
import random

tf.random.set_seed(111111)
np.random.seed(111111)
random.seed(111111)

class Squash(Layer):

    def __init__(self, eps=10e-21, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, s):
        n = tf.norm(s,axis=-1,keepdims=True)
        return (1 - 1/(tf.math.exp(n)+self.eps))*(s/(n+self.eps))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape
    

class PrimaryCaps(Layer):
    def __init__(self, outF, inF, K, N, D, s=1, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.outF = outF
        self.inF = inF
        self.K = K
        self.N = N
        self.D = D
        self.s = s
        
    def get_config(self):
        config = {
            'K': self.K,
            'N': self.N,
            'D': self.D
        }
        base_config = super(PrimaryCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def build(self, input_shape):   
        depth = int((self.outF) / self.inF)
        self.DW_Conv2D = DepthwiseConv2D(kernel_size=self.K, strides=self.s, padding='valid', depth_multiplier=depth, activation='linear')
        self.built = True
    
    def call(self, inputs):
        inputs= inputs
        x = self.DW_Conv2D(inputs)  
        x = Reshape((self.N, self.D))(x)
        x = Squash()(x)
        
        return x
    


class MHACaps(Layer):
    def __init__(self, N, D, heads, value_size=None, key_size=None, kernel_initializer='he_normal', **kwargs):
        super(MHACaps, self).__init__(**kwargs)
        self.N = N
        self.D = D
        self.heads = heads # 多头的数目
        self.value_size = value_size or (D // heads) # v头的大小
        self.key_size = key_size or self.value_size # qk每个头的大小  
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        input_N = input_shape[-2]
        input_D = input_shape[-1]
        self.param = {}
        for i in range(self.heads):
            W = 'W' + str(i+1)
            Wq = 'Wq' + str(i+1)
            Wk = 'Wk' + str(i+1)
            b = 'b' + str(i+1)
            self.param[W] = self.add_weight(shape=[self.N, input_N, input_D, self.value_size],initializer=self.kernel_initializer,name=W)
            self.param[Wq]=self.add_weight(shape=[self.N, input_N, input_D, self.key_size],initializer=self.kernel_initializer,name=Wq)
            self.param[Wk]=self.add_weight(shape=[self.N, input_N, input_D, self.key_size],initializer=self.kernel_initializer,name=Wk)
            self.param[b] = self.add_weight(shape=[self.N, input_N, 1], initializer=tf.zeros_initializer(), name=b)
            
        self.dense = Dense(units=self.D, kernel_initializer=self.kernel_initializer)
        
        self.built = True
    
    def call(self, inputs, training=None):

        S = []
        for i in range(self.heads):
            W = self.param['W' + str(i+1)]
            Wq = self.param['Wq' + str(i+1)]
            Wk =self.param['Wk' + str(i+1)]
            b = self.param['b' + str(i+1)]
            
            u = tf.einsum('...ji,kjiz->...kjz', inputs, W)    # u shape=(None, N, input_N, self.value_size)
            q = tf.einsum('...ji,kjiz->...kjz', inputs, Wq)
            k = tf.einsum('...ji,kjiz->...kjz', inputs, Wk)
            
            c = tf.einsum('...ij,...kj->...i', q, k)[...,None]  # c shape=(None, N, input_N, 1)
            
            c = c/tf.sqrt(tf.cast(self.value_size, tf.float32))
            c = tf.nn.softmax(c, axis=1)                             # c shape=(None,N,input_N,1) -> (None,j,i,1)
            c = c + b
            s = tf.reduce_sum(tf.multiply(u, c),axis=-2)             # s shape=(None,N,self.value_size)
            
            S.append(s)
        if self.heads > 1: 
            o = concatenate(S)
        else:
            o = S[0]
        s = self.dense(o)
        v = squash(s)
        return v
    
    def compute_output_shape(self, input_shape):
        return (None, self.N, self.D)

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D
        }
        base_config = super(MHACaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class FCCaps(Layer):
    def __init__(self, N, D, kernel_initializer='he_normal', **kwargs):
        super(FCCaps, self).__init__(**kwargs)
        self.N = N
        self.D = D
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        input_N = input_shape[-2]
        input_D = input_shape[-1]

        self.W = self.add_weight(shape=[self.N, input_N, input_D, self.D],initializer=self.kernel_initializer,name='W')
        self.b = self.add_weight(shape=[self.N, input_N, 1], initializer=tf.zeros_initializer(), name='b')
        self.built = True
    
    def call(self, inputs, training=None):
        
        u = tf.einsum('...ji,kjiz->...kjz',inputs,self.W)    # u shape=(None, N, input_N, D)  
        #c = tf.einsum('...ij,...kj->...i', u, u)[...,None]        # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        #c = c/tf.sqrt(tf.cast(self.D, tf.float32))
        #c = tf.nn.softmax(c, axis=1)                             # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        #c = c + self.b
        
        #s = tf.reduce_sum(tf.multiply(u, c),axis=-2)             # s shape=(None,N,D)
        s = tf.reduce_sum(u,axis=-2)             # s shape=(None,N,D)
        v = Squash()(s)       # v shape=(None,N,D)
        
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D
        }
        base_config = super(FCCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class PrimaryCaps_H(tf.keras.layers.Layer):
    def __init__(self, C, L, k, s, padding='VALID', **kwargs):
        super(PrimaryCaps_H, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.k = k
        self.s = s
        self.padding = padding
        
    def build(self, input_shape):    
        self.kernel = self.add_weight(shape=(self.k, self.k, input_shape[-1], self.C*self.L), initializer='glorot_uniform', name='kernel')
        self.biases = self.add_weight(shape=(self.C,self.L), initializer='zeros', name='biases')
        self.built = True
    
    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self.kernel, self.s, self.padding)
        H,W = x.shape[1:3]
        x = tf.keras.layers.Reshape((H, W, self.C, self.L))(x)
        x /= self.C
        x += self.biases
        x = squash(x)      
        return x
    
    def compute_output_shape(self, input_shape):
        H,W = input_shape.shape[1:3]
        return (None, (H - self.k)/self.s + 1, (W - self.k)/self.s + 1, self.C, self.L)

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'k': self.k,
            's': self.s
        }
        base_config = super(PrimaryCaps_H, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DigitCaps(tf.keras.layers.Layer):
    def __init__(self, C, L, routing=None, kernel_initializer='glorot_uniform', **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.routing = routing
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None,H,W,input_C,input_L]"
        H = input_shape[-4]
        W = input_shape[-3]
        input_C = input_shape[-2]
        input_L = input_shape[-1]

        self.W = self.add_weight(shape=[H*W*input_C, input_L, self.L*self.C], initializer=self.kernel_initializer, name='W')
        self.biases = self.add_weight(shape=[self.C,self.L], initializer='zeros', name='biases')
        self.built = True
    
    def call(self, inputs):
        H,W,input_C,input_L = inputs.shape[1:]          # input shape=(None,H,W,input_C,input_L)
        x = tf.reshape(inputs,(-1, H*W*input_C, input_L)) #     x shape=(None,H*W*input_C,input_L)
        
        u = tf.einsum('...ji,jik->...jk', x, self.W)      #     u shape=(None,H*W*input_C,C*L)
        u = tf.reshape(u,(-1, H*W*input_C, self.C, self.L))#     u shape=(None,H*W*input_C,C,L)
        
        if self.routing:
            #Hinton's routing
            b = tf.zeros(tf.shape(u)[:-1])[...,None]                       # b shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
            for r in range(self.routing):
                c = tf.nn.softmax(b,axis=2)                                # c shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
                s = tf.reduce_sum(tf.multiply(u,c),axis=1,keepdims=True)   # s shape=(None,1,C,L)
                s += self.biases       
                v = squash(s)                                              # v shape=(None,1,C,L)
                if r < self.routing-1:
                    b += tf.reduce_sum(tf.multiply(u, v), axis=-1, keepdims=True)
            v = v[:,0,...]      # v shape=(None,C,L)
        else:
            s = tf.reduce_sum(u, axis=1, keepdims=True) 
            s += self.biases
            v = squash(s)
            #v = Squash()(s)
            v = v[:,0,...]
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'routing': self.routing
        }
        base_config = super(DigitCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ARCaps(Layer):
    def __init__(self, N, D, kernel_initializer='he_normal', **kwargs):
        super(ARCaps, self).__init__(**kwargs)
        self.N = N
        self.D = D
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        H = input_shape[-4]
        W = input_shape[-3]
        input_N = input_shape[-2]
        input_D = input_shape[-1]
        
        self.W = self.add_weight(shape=[H*W*input_N, input_D, self.D*self.N], initializer=self.kernel_initializer, name='W')
        self.conv3d = tf.keras.layers.Conv3D(input_N, (1,1,self.D), padding='valid', kernel_initializer=self.kernel_initializer)
        self.built = True
    
    def call(self, inputs, training=None):
        H,W,input_N,input_D = inputs.shape[1:]          # input shape=(None,H,W,input_C,input_L)
        x = tf.reshape(inputs,(-1, H*W*input_N, input_D)) #     x shape=(None,H*W*input_C,input_L)
        
        u = tf.einsum('...ji,jik->...jk', x, self.W)      #     u shape=(None,H*W*input_C,C*L)
        u = tf.reshape(u,(-1, H*W*input_N, self.N, self.D))#     u shape=(None,H*W*input_C,C,L)
        temp = []
        for n in range(self.N):
            b = self.conv3d(tf.transpose(tf.reshape(u[:,:,n,:], (-1, H, W, input_N, self.D)), (0,1,2,4,3)))
            c = tf.nn.softmax(b, axis=-1)                            
            temp.append(tf.reshape(c, (-1, H*W*input_N, 1)))
        c = tf.stack(temp,-2)
        s = tf.reduce_sum(tf.multiply(u,c),axis=1,keepdims=True)
        v = Squash()(s)       # v shape=(None,N,D)
        v = v[:,0,...]
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.N, self.D)

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D
        }
        base_config = super(ARCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))