import pandas as pd
import random
#데이터 생성 코드

#df = pd.DataFrame()
#for i in range(0,5):
# list_box = []
# for k in range(0,200):
#  ran_num = random.randint(1,200)
#  list_box.append(ran_num)
# df[i+2000] = list_box
#df = df.T
#df.to_csv("data.csv", index = True)

#index '2001' 데이터(행)의 평균보다 큰 값의 수와 2003 데이터의 평균보다 작은 값의 수를 더하시오

df = pd.read_csv('data.csv', index_col = 0)
#print(df.head())
df = df.T
value_2001 = df[2001] > df[2001].mean()
value_2003 = df[2003] < df[2003].mean()

v1 = value_2001.sum()
v2 = value_2003.sum()
print(v1+v2)
