#-*- coding:utf-8 -*-

# x=[1,2,3]
# y=[1,2,3]
# x=5일때, y=?

import tensorflow as tf 

# 1. 입력 데이터 정의  
x_data = [1,2,3]
y_data = [1,2,3]

# 2. 예측 공식 생각 
# y = W*X+b

# 3. 입력 데이터를 위한 placeholder 선언  
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 4. 최적화 함수를 위한 변수 선언  
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 5. 예측 공식 정의, (행렬이 아니니 수식으로만 표현 )
hypothesis = W*X+b

# 6. cost 함수 정의 
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# 7. 최적화 함수 선택 (경사 하강법 )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# 8. 최적화 결정 함수 선택 (cost가 최저가 될때 )
train_op = optimizer.minimize(cost)

# 9. 세션 준비, 변수 초기화  
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# 10. 학습 시작 
	for step in range(100):
		_, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})
		print (step, cost_val, sess.run(W), sess.run(b))
	# 11. 예측치 출력 
	print ("X:5, Y:", sess.run(hypothesis, feed_dict={X:5}))



