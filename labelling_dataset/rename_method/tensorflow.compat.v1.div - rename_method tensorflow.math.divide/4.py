# -*- coding: utf-8 -*-
"""
step03_tensorboard.py

 - Tensorboard & 사칙연산 함수
 
 1. Tensorboard : tensorflow 시각화 도구
 2. 사칙연산 함수
    tf.add(x, y, name) -> a + b
    tf.subtract(x, y, name)
    tf.div(x, y, name)
    tf.multiply(x, y, name)
"""

import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용

# tensorboard 초기화
tf.reset_default_graph()

# 상수 정의
x = tf.constant(1, name = "x")
y = tf.constant(2, name = "y")

# 사칙연산 : 식 정의
# name : 공백, 특수문자, 한글 사용 불가
a = tf.add(x, y, name = "a")
b = tf.multiply(a, 6, name = "b")

c = tf.subtract(20, 10, name = "c")
d = tf.div(c, 2, name = "d")

g = tf.add(b, d, name = "g")
h = tf.multiply(g, d, name = "h")


sess = tf.Session()
sess.close()
with tf.Session() as sess: # close()가 필요 없음
    # 객체 공간
    print("h = ", sess.run(h)) # h =  115
    tf.summary.merge_all() # tensor 모으는 역할
    writer = tf.summary.FileWriter("C:/ITWILL/6_Tensorflow/graph", sess.graph)
    writer.close()
    
    
# 객체 소멸






















