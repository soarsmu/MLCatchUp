# -*- coding: utf-8 -*-
"""

    - if , while 

"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

# 1. if문
x = tf.constant(10)

def true_fn():
    return tf.multiply(x , 10)

def false_fn():
    return tf.add(x , 10)

y = tf.cond( x > 100 ,true_fn , false_fn )

'''

pred : 조건식
true_fn : 조건식이 참인 경우 수행하는 인수가 없는 함수
false_fn : 조건식이 거짓인 경우 수행하는 인수가 없는 함수

'''

# 2. while문
i = tf.constant( 0 )
def cond(i):
    return tf.less(i , 100)

def body(i) :
    return tf.add(i,1)
 
loop = tf.while_loop( cond , body , (i,))

'''
cond : 조건식 ( 호출 가능한 함수) 
body : 반복문(호출 가능한 함수)
loop_vars : 반복 변수 (tuple or list)
'''


with tf.Session() as sess:
    print(sess.run(y))
    print("loop = ", sess.run(loop))


























