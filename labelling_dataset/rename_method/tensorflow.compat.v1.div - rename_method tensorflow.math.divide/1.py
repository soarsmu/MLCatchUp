''' * Stain-Color Normalization by using Deep Convolutional GMM (DCGMM).
    * VCA group, Eindhoen University of Technology.
    * Ref: Zanjani F.G., Zinger S., Bejnordi B.E., van der Laak J. AWM, de With P. H.N., "Histopathology Stain-Color Normalization Using Deep Generative Models", (2018).'''
    
import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

def dataset_tf(dataset):
  return dataset

def GMM_M_Step(X, Gama, ClusterNo, name='GMM_Statistics', **kwargs):

    D, h, s = tf.split(X, [1,1,1], axis=3)
    
    WXd = tf.multiply(Gama, tf.tile(D ,[1,1,1,ClusterNo]))
    WXa = tf.multiply(Gama, tf.tile(h ,[1,1,1,ClusterNo]))
    WXb = tf.multiply(Gama, tf.tile(s ,[1,1,1,ClusterNo]))
    
    S = tf.reduce_sum(tf.reduce_sum(Gama, axis=1), axis=1)
    S = tf.add(S, tensorflow.keras.backend.epsilon())
    S = tf.reshape(S,[1, ClusterNo])
    
    M_d = tf.div(tf.reduce_sum(tf.reduce_sum(WXd, axis=1), axis=1) , S)
    M_a = tf.div(tf.reduce_sum(tf.reduce_sum(WXa, axis=1), axis=1) , S)
    M_b = tf.div(tf.reduce_sum(tf.reduce_sum(WXb, axis=1), axis=1) , S)
    
    Mu = tf.split(tf.concat([M_d, M_a, M_b],axis=0), ClusterNo, 1)  
    
    Norm_d = tf.squared_difference(D, tf.reshape(M_d,[1, ClusterNo]))
    Norm_h = tf.squared_difference(h, tf.reshape(M_a,[1, ClusterNo]))
    Norm_s = tf.squared_difference(s, tf.reshape(M_b,[1, ClusterNo]))
    
    WSd = tf.multiply(Gama, Norm_d)
    WSh = tf.multiply(Gama, Norm_h)
    WSs = tf.multiply(Gama, Norm_s)
    
    S_d = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSd, axis=1), axis=1) , S))
    S_h = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSh, axis=1), axis=1) , S))
    S_s = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSs, axis=1), axis=1) , S))
   

