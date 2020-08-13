"""
if, while 형식
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. if
def true_fn():
    return tf.multiply(x, 10)

def false_fn():
    return tf.add(x, 10)

x = tf.constant(10)
y = tf.cond(x > 100, true_fn, false_fn)
'''
pred : 조건식
true_fn : 조건식이 참인 경우 수행하는 함수(인수 없음)
false_fn : 조건식이 거짓인 경우 수행하는 함수(인수 없음)
'''

# 2. while
i = tf.constant(0)  # i = 0 : 반복변수

def cond(i):
    return tf.less(i, 100)  # i < 100

def body(i):
    return tf.add(i, 1)  # i += 1

loop = tf.while_loop(cond=cond, body=body, loop_vars=(i,))
'''
cond : 조건식(호출가능한 함수)
body : 반복식(호출가능한 함수)
loop_vars : 반복변수(tuple or list)
'''

sess = tf.Session()
print('y =', sess.run(y))
print('loop =', sess.run(loop))