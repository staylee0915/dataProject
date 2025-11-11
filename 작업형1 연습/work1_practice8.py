import pandas as pd
import numpy as np
# 결측값을 가진 데이터는 바로 뒤에 있는 값으로 대체한 후 (바로뒤가 결측값이라면 뒤에있는 데이터 중 가장 가까운 값)
# city와 f2 컬럼 기준으로 그룹합을 계산
# views가 세번쨰로 큰 city의 이름

df = pd.read_csv('members.csv')

#print(df.head())
#print(df.isnull().sum())

#fillna(method = 'bfill')을 이용해서 결측값을 바로 뒤에 있는 가까운 값으로 채운다.
df = df.bfill()
#print(df)
#print(df.isnull().sum())

#2. 그룹합 구하기
g_df = df.groupby(['city','f2']).sum().reset_index()
#print(g_df)
# 세번쨰로 큰 city
g_df = g_df.sort_values('views',ascending = False)
print(g_df['city'].iloc[2])