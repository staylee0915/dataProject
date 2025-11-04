# data : members.csv
# 한개의 셀만 사용해서 문제풀이
# 문제1. f1컬럼의 결측치는 중앙값으로 대체
# 나머지 결측치가 있는 데이터(행)을 모두 제거,
# 앞에서 부터 70%의 데이터 중 views 컬럼의 3사분위 수에서 1사분위 수를 뺸값을 구하라
# 단 데이터 70% 지점은 정수형 (int) 변환

import pandas as pd

df = pd.read_csv('members.csv') #ipynb와 다르게 실제 경로 지정 필요.

#csv에 포함된 데이터 확인.
#print(df.head())

#1. 결측값을 중앙값 대체
#항목별 결측값 확인 (sum)을 확인해서 확인
#print(df.isnull().sum())
df['f1'] = df['f1'].fillna(value=df['f1'].median())
#print(df.isnull().sum())
#print("--- 1번 끝 ---")

#2. 나머지 결측치가 있는 값 모두 제거
#print(df.shape)
#나머지 결측값 모두 제거하여 다시 대입.
df = df.dropna() 
#print(df.shape)

#print("--- 2번 끝 ---")
#앞에서 부터 70% 데이터중 views 컬럼의 3사분위 수에서 1사분위 수를 뺸값
#70프로 지점은 정수형으로 반환

#70프로지점의 정수형.
df_70 = df.iloc[0:int(df.shape[0]*0.7)]
#70프로 지점의 정수형은 int(len(df)*0.7)로 구해도 가능함.
#print(df_70.shape)

#70프로의 데이터 중 views 컬럼의 3사분위 수에서 1사분위수를 뺀값.
print(df_70['views'].quantile(0.75) - df_70['views'].quantile(0.25))