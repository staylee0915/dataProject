# 1. views 컬럼에 결측치가 있는 데이터(행)을 삭제
# 2. f3 컬럼의 결측치는 0, silver는 1, gold는 2, vip는 3으로 변환한 후 f3컬럼의 총 합을 정수형으로 출력

import pandas as pd

df = pd.read_csv('members.csv')
print(df.head())

# 결측값 확인
#print(df.isnull().sum())
df = df.dropna(subset=['views'])
#print(df.shape) # 행이 제거 되었음을 확인, subset[], axis = 0은 결측값이 있는 행을 제거.
df['f3'] = df['f3'].fillna(0)
#위 구문은 아래와 같이도 사용가능
#import numpy as np
#df['f3'] = df['f3'].replace(np.nan,0)

df['f3'] = df['f3'].replace({'silver' : 1, 'gold' : 2, 'vip' : 3}).astype(int)
#print(df['f3'])
print(df['f3'].sum())