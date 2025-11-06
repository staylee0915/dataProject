#1. 자료형이 type인 object 컬럼은 삭제하고 결측치는 0으로 대체
#2. 행단dnl로 합한 다음 그 값이 3000보다 큰 값의 데이터 수를 구하라

import pandas as pd

df = pd.read_csv('members.csv')


#print(df.dtypes)

#데이터 타입이 object인 컬럼 선택
cols = df.select_dtypes(include = 'object').columns
#자료형이 object인 컬럼 삭제
df = df.drop(cols,axis=1)

#print(df.head())
#결측값은 0으로 대체
df = df.fillna(0)
#print(df.head())
#각 행의 병렬 합.
#print(df.sum()) -> 이와같이 sum을 찍으면 column별로 더해진다. 따라서 변환필요

#열단위로 더하기 위해 전치
df = df.T
#print(df.head())
df.head()

#각 행의 합이 3000보다 큰 것을 확인
#print(df.sum()>3000)

print(sum(df.sum()>3000)) # t는 1 f는 0이므로 3000보다 큰 것들의 합은 73이 나온다.