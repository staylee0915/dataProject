import pandas as pd

#구독월별로 데이터 개수를 구한 뒤
#가장 작은 구독수가 잇는 월을 구하라
df = pd.read_csv('members.csv')

#print(df.dtypes)
#print(df.head())
df['subscribed'] = pd.to_datetime(df['subscribed'])
#print(df.dtypes)

#연과 월을 구하기
df['year'] = df['subscribed'].dt.year
df['month'] = df['subscribed'].dt.month

#월로 그룹화 하여 데이터 갯수 구하기
#print(df.head())
#month를 기준으로 그룹하고 month의 갯수를 센다.
df = df.groupby('month').count()
#month를 기준으로 그룹화된 df를 subscribed 컬럼을 기준으로 오름차순 정렬하고, 그 인덱스를 가져온다
idx = df.sort_values('subscribed',ascending = True).index
#11이 가장 작은 구독수가 있는 월이다.
print(idx[0])