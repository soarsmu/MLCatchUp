import tensorflow as tf
import numpy as np

#  create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

# create tensorflow structure start
Weights = tf.Variable(tf.compat.v1.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# 预测的Y与真实y的差别
loss = tf.compat.v1.reduce_mean(tf.square(y-y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.compat.v1.initialize_all_variables()
# create tensorflow structure end

sess = tf.compat.v1.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))