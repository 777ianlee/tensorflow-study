#-*- coding: utf-8 -*-

# X와 Y의 상관 관계를 분석하는 기초적인 선형 회귀 모델 (linear regression)
# X = [1,2,3]
# Y = [1,2,3]

import tensorflow as tf


# hypothesis = W * X + b

# W:Weight(가중치), shape=1, -1.0에서 1.0 사이의 균등 분포 난수 발생
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b:bais (편향), shape=1, -1.0에서 1.0사이의 균등 분포 난수 발생
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# X,Y 좌표값
x_data = [1,2,3]
y_data = [1,2,3]

# 좌표를 입력 받을 placeholder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 손실 값 = 모든 좌표에 대한 평균값(square(hypothesis-Y))
hypothesis = W*X+b
# 학습이라고 부르는 건 아래 cost값을 최소화하는 W와 b값을 구하는 것이다.
# (W,b는 학습에서 구해지는 값이기 때문에 Variable로 선언한다.)
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# tensorflow에서 제공하는 GradientDescent(경사하강법) 함수를 써서 w,b를 구하자.
# learning_rate=학습률, hyperparameter, 너무 크면 최적값을 지나치고, 너무 작으면 오래 걸린다.
# 머링러닝에서는 이 hyperparameter를 튜닝하는 게 큰 과제이다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# cost가 최저인 값을 알려주는 trainer
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# 최적화를 100번 수행
	for step in range(100):
		# train_op와 cost를 계산하는 데, X에 x_data를 Y에 y_data를 입력 값으로 넣는다.
		# train_op의 반환값을 _(무시)로 받고 cost 함수의 반환 값을 cost_val로 받는다.
		_, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})
		# 아래와 같이 두개를 나눠서 호출해도 된다.
		# sess.run(train_op, feed_dict={X:x_data, Y:y_data})
		# cost_val = sess.run(cost, feed_dict={X:x_data, Y:y_data})
		print(step, cost_val, sess.run(W), sess.run(b))

	print ("X:5, Y:", sess.run(hypothesis, feed_dict={X:5}))
	print ("X:2.5, Y:", sess.run(hypothesis, feed_dict={X:2.5}))






