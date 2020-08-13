# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:09:22 2020

step09_tf_logic.py

- if, while 형식 

"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

# 1. if문 
x = tf.constant(10)

def true_fn() :
    return tf.multiply(x, 10) #x*10

def false_fn():
    return tf.add(x, 10) #x+10

#y = tf.cond(pred, true_fn, false_fn)
y = tf.cond( x > 100 , true_fn, false_fn) #false 

'''
pred :조건식
true_fn:조건식이 참인 경우 수행하는 함수. 인수없는 함수 
false_fn:조건식이 거짓인 경우 수행하는 함수. 인수가 없는 함수
'''

# while
i = tf.constant(0) #i=0 :반복변수

def cond(i) :
    return tf.less(i, 100) # i < 100  100 전까지 반복수행 

def body(i):
    return tf.add(i,1) # cond만족하는 것이 끝나면 i = i+1 까지 반복수행

loop = tf.while_loop(cond, body, (i,)) # tuple (i,) list [i]

'''
cond : 조건식(호출가능한 함수)
body : 반복문(호출가능한 함수)
loop_vars : 반복변수(tuple or list)
'''
sess = tf.Session()

print("y=", sess.run(y)) #y= 20
print("loop=", sess.run(loop)) #loop= 100