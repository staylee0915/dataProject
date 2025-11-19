import pandas as pd
'''
#통계값 확인 
import pandas as pd
import numpy as np

#데이터 탐색
1. 데이터 갯수 확인
pd.shape
2. 데이터 타입 확인
pd.info
3. 데이터 샘플확인 및 최대 최소 평균값 등 수치 확인
pd.describe() // 수치형
pd.describe(include = 'object') // 범주형
4. 결측값 확인
pd.isnull().sum()

#데이터 전처리
1. 타겟값 분리
target = df[''].pop(df[''])

2. 결측값 채우기
df.fillna(train[].fillna(train[].median()))
df.fillna(train[].fillna(train[].mode()[0]))
df.fillna(train[].fillna(train[].min())

test데이터에대해서도 결측값을 채워줘야 함.

#인코딩
범주형 데이터를 숫자형으로 변환
object컬럼들을 가져와서 변환해 주어야 함

cols = list(train.columns[train.dtypes == object])
train과 test의 데이터 셋을 합쳐셔 하나로 만들어줌
df = pd.concat([train,test])

인코딩은 레이블 인코딩과 원핫 인코딩이 대표적으로 많이 쓰임
원핫 인코딩의 경우 컬럼의 unique수가 너무 많으면 데이터 양이 많아 지기에 이경우에는 레이블 인코딩을 씀

1. 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cols :
    df[col] = le.fit_transform(df[col])

(train test 셋 제작)
train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):].copy()

2. 원핫 인코딩
train과 test의 컬럼이 동일할 경우 
만약 컬럼의 unique 수가 너무 많으면 특정 컬럼을 drop 한 후 원 핫 인코딩을 진행해야함

df = df.drop('item', axis = 1)
(train test셋 제작)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

기본 (카테고리 같음)

#데이터 분할

from sklearn.model_selection import train_test_split

xt,xv,yt,yv = train_test_split(
    train,
    target,
    test_size = 0.2,
    random_state = 0
    #
    #층화추출
)
#학습모델 선정 (회귀, 분류)
(회귀))
from sklearn.linear_model import LinearRegression
// 분류일 경우 import LogisticRegression (로지스틱 회귀는 분류문제에 사용하는 것.)

from sklearn.emsemble import RandomForestRegressor
// 분류문제일 경우 import RandomForestClassifier

#학습
rf = RandomForestRegressor()
 - 훈련
rf.fit(xt,yt)
y_pred = rf.predict(xv)

#검증
target의 validation 셋과 학습의 결과 발견된 예측값으로 result 생성
result = root_mean_squared_error(yv,y_pred)
print(result)


#파일로 출력
submit = pd.DataFrame({'pred':pred})
submit.to_csv("result.csv",index = False)

print(pd.read_csv('result.csv'))
#최종 라인 확인

import sklearn.metrics import root_mean_squared_error
mean_squred_error
mean_Absolute_error
r2_score

'''