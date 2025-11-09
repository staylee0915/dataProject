import pandas as pd

#데이터에서 iqr을 이용해 biews 컬럼의 이상치를 찾고, 이상치 데이터수를 구하시오

df = pd.read_csv('members.csv')

#iqr의 이상치는 1.5iqr로 구하며 iqr은 3사분위수 - 1사분위수로 구한다.
# q1 - 1.5iqr 보다 작거나 q3 +1.5iqr보다 크면 이상치로 구분한다.

q1 = df['views'].quantile(0.25)
q3 = df['views'].quantile(0.75)

iqr = q3-q1

outlier_l = q1-1.5*iqr
outlier_r = q3+1.5*iqr

answer = sum(df['views']<outlier_l) + sum(df['views']>outlier_r)

print(answer)
