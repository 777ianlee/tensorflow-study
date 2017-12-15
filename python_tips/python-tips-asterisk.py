#-*- coding: utf-8 -*-

# * 활용법

# 길이 100의 제로 값 리스트 초기
zeros_list = [0]*10
print ('zeros_list:', zeros_list)

# 길이 100의 제로값 튜플 선언 , 튜플임을 인식시키기 위해 숫자 다음에 , 추가 
zeros_tuple = (0,)*10
print ('zeros_tuble:', zeros_tuple)

#리스트 3배 확장 후 연산, enumerate는 offset과 객체를 반환 
vector_list = [[1,2,3]]
for i,vector in enumerate(vector_list*3):
	print ("{0} scalar product of vector:{1}".format((i+1), [(i+1)*e for e in vector]))

