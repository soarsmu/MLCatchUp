# importing the tensorflow package
#import tensorflow as tf
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""
print(tf.reduce_sum(tf.random.normal([10, 10])))

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
"""
val = tf.print(tf.Variable(tf.random_normal([1])), [tf.Variable(tf.random_normal([1]))] )

print(tf.random_normal([1]))

logits = [5.0, 2.0, 1.5]
exps = [np.exp(i) for i in logits]
sume = sum(exps)
softmax = [j/sume for j in exps]

print()
print(logits, exps)
print(sume, softmax)
print(sum(softmax))


e = np.exp(1)
print(e**(-100) > 0)

x = tf.constant([[1, 1, 1], [1, 1, 1]])
a = tf.reduce_sum(x)
b = tf.reduce_sum(x, [1]) # 6


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    a_value = sess.run(a)
    b_value = sess.run(b)
    print(a_value)
    print(b_value)

