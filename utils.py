import tensorflow as tf
from tensorflow.keras import backend as K

def squash(s):
    n = tf.norm(s, axis=-1,keepdims=True)
    return tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), s)

def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)