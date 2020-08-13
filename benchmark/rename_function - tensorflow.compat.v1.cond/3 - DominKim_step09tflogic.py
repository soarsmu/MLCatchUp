# -*- coding: utf-8 -*-
"""
step09_tf_logic.py
 - if, while 형식
"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

# 1. if문
x = tf.constant(10) # x = 10

def true_fn():
    return tf.multiply(x, 10) # x * 10

def false_fn():
    return tf.add(x, 10) # x + 10

y = tf.cond(x > 100, true_fn, false_fn) # false
'''
pred = 조건식
true_fn  : 조건식이 참인 경우 수행하는 함수
false_fn : 조건식이 거짓인 경우 수행하는 함수(인수 없음)
'''

# 2. while 
i = tf.constant(0) # i = 0 : 반복변수

def cond(i):
    return tf.less(i, 100) # i < 100

def body(i):
    return tf.add(i, 1) # i += 1

loop = tf.while_loop(cond, body, (i,))
'''
cond : 조건식(호출 가능한 함수)
body : 반복문(호출 가능한 함수)
loop_vars : 반복변수(tuple or list)
'''

sess = tf.Session()
print("y =", sess.run(y)) # y = 20
print("loop =", sess.run(loop)) # loop = 100

sess.close()