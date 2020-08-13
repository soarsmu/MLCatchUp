# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:10:02 2020

@author: user

-Tensorboard & 사칙연산 함수 
1. Tensorboard: Tensorflow 의 시각화 도구 
2. 사칙연산 함수 
    tf.add(x, y, name) -> a + b 
    tf.subtract(x, y, name)
    tf.div(x, y, name)
    tf.multiply(x, y, name)

"""
import tensorflow.compat.v1 as tf #ver 1.x
tf.disable_v2_behavior() #ver 2.x 사용

#상수 정의 
x = tf.constant(1, name='x')
y = tf.constant(2, name='y')

#사칙연산 : 식 정의 
#name지정시 쓸수 없는 것 : 공백, 특수문자, 한글 
#name지정시 사용 가능 : _ 

a = tf.add(1,2,name='a')
b = tf.multiply(a, 6, name='b')
c = tf.subtract(10,20,name='c')
d = tf.div(c, 2, name='d')

g = tf.add(b,d, name='g')
h = tf.multiply(g, d, name='h')

#sess = tf.Session() # 객체 만들고, 나중에 객체 닫기 
with tf.Session() as sess :  # 들여쓰기 객체 공간, 내어쓰기 자동 객체 소멸 so, with 사용
 
   print("h=", sess.run(h)) 
   print("b=", sess.run(b)) 
   print("c=", sess.run(c)) 
   print("d=", sess.run(d)) 
   print("g=", sess.run(g)) 
   print("a=", sess.run(a))
   
   tf.summary.merge_all() # h= 115
   writer = tf.summary.FileWriter("C:/ITWILL/6_Tensorflow/graph", sess.graph)
   #<tensorflow.python.summary.writer.writer.FileWriter at 0x19d45edb8c8>
   print()
   writer.close()
   
   
