
#1. 데이터 결측치가 30% 이상이 되는 컬럼을 찾고, 해당 컬럼에 결측치가 있는 데이터(행)삭제
#2. 그리고 30% 미만, 20% 이상인 결측치가 있는 컬럼은 최빈값으로 값을 대체
#3. f3 컬럼의 gold 값을 가진 데이터 수 출력.

import pandas as pd

df = pd.read_csv('members.csv')
#결측값 비율
#print(df.isnull().sum()/len(df))
# f1 삭제
# f3 최빈값

#print(df.shape)
df = df.dropna(subset=['f1']) #f1의 결측값 삭제.
#print(df.shape)

#print(df['f3'].mode()[0]) //최빈값 확인.
df['f3'] = df['f3'].fillna(df['f3'].mode()[0])
#print(df.head())

print(sum(df['f3'] == 'gold'))

