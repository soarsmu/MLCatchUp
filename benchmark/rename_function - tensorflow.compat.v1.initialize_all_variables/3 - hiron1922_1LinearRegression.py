import numpy as np
import tensorflow.compat.v1 as tf # 降低版本

tf.disable_v2_behavior()

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tf's structure start #
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases  # predict the value of y

loss = tf.reduce_mean(tf.square(y - y_data))  # loss function
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 优化器使用梯度下降算法实现 0.5是学习效率
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# create tf's structure end #

sess = tf.compat.v1.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
