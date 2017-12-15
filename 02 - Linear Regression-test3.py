#-*- coding:utf-8 -*-

# 문제
# x=[1,2,3]
# y=[1,2,3] 일때,
# x=5이면 y=?

import tensorflow as tf

# 0. 입력 데이터 선언
x_data = [1,2,3]
y_data = [1,2,3]

# 1. 문제를 그래프로 그려보고 해결 수식을 만든다
# Y = W*X +b

# 2. 필요한 데이터 입력을 위해 placeholder를 만든다
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 3. 나머지는 최적화 함수를 위한 변수로 선언한다.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 4. 예측 그래프를 만든다. (여기서는 행렬이 아니니 수식만)
hypothesis = W*X+b

# 5. 변수가 달라지는 모습을 그래프로 그려보고 cost 함수를 만든다 (여기서는 모든 기울기와 편차)
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# 6. cost 함수가 2차 방정식이니 최적화 함수로 GradientDescentAlgorithm class를 선택하다
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# 7. 최적화 함수가 끝나는 조건은 cost가 최저값을 찾았을 때이다.
train_op = optimizer.minimize(cost)

# 8. 세션과 변수를 초기화
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# 9. 학습 시작
	for i in range(100):
		_, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})
		print ('[{0}] cost:{1}, W:{2}, b:{3}'.format(i, cost_val, sess.run(W), sess.run(b)))

	print ('X:5, Y=', sess.run(hypothesis, feed_dict={X:5}))

