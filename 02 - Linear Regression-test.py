#-*- coding: utf-8 -*-

# X와 Y의 상관 관계를 분석하는 기초적인 선형 회귀 모델(linear regression)
# X=[1,2,3]
# Y=[1,2,3]

import tensorflow as tf

# 1. 데이터 입력 
x_data = [1,2,3]
y_data = [1,2,3]

# 2. 예측 공식을 만든다.
# hypothesis = W*X+b

# 3. 데이터 입력을 위한 placeholder 선언한다.
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 4. 최적화 함수가 계산해줬으면 하는 변수를 선언한다.
# 초기 값은 임의로 설정한다.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 5. 예측 공신을 만든다.
hypothesis = W*X+b 

# 6. 비용 함수를 만든다. 
cost = tf.reduce_mean(tf.square(Y-hypothesis))

# 7. 비용 최적화 함수를 선택한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

# 6. session을  만들고  
with tf.Session() as sess:
	# 7. 변수를 초기화 하고  
	sess.run(tf.global_variables_initializer())
	# 8. 반복하면서  
	for step in range(100):
		# 9. 학습한다. 
		_, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})
		print (step, cost_val, sess.run(W), sess.run(b))

	# 10. 최적화된 변수를 이용하여 예측치를 출력한다.
	print ("X:5, Y:", sess.run(hypothesis, feed_dict={X:5}))
	print ("X:2.5 Y:", sess.run(hypothesis, feed_dict={X:2.5}))

