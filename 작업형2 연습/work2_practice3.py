#평가 r-squared, mae, mse, rmse, rmsle, mape
#target  = price
# 수험번호.ㅊㄴㅍ
#id 와 price를 생성

import pandas as pd

train = pd.read_csv('content/practice3/train.csv')
test = pd.read_csv('content/practice3/test.csv')

#탐색적 분석
print(train.shape, test.shape)

print(train.info())
print(test.info())

print(train.describe())
print(test.describe())

#결측값 확인
print(train.isnull().sum())
print(test.isnull().sum())

#결측값 처리
print(train.isnull())
#타겟 확인
cols = train.select_dtypes(include='object').columns
target = train.pop('price')
print(train.shape)
print(target.describe())

#결측값 삭제행
#last_review                       1999
#reviews_per_month                 1999
#name
#hostname
#object 3개, float 1개

#결측치 처리 필요.
print(train.nunique())
#name, host_name, last_review , hostid삭제
#reviews_per_month -> 0
dropcols = ['name','host_name','last_review','host_id']
train = train.drop(dropcols,axis = 1)
test = test.drop(dropcols,axis = 1)
print(train.shape)

train['reviews_per_month'] = train['reviews_per_month'].fillna(0)
test['reviews_per_month'] = test['reviews_per_month'].fillna(0)
print(train.isnull().sum())

train = train.drop('id',axis = 1)

test_id = test.pop('id')
print(train.info())
from sklearn.preprocessing import LabelEncoder

cols = ['neighbourhood','room_type','neighbourhood_group']

for col in cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col]) # train data에 적용
    test[col] = le.transform(test[col]) # test 데이터에 적용

print(train[cols])

#데이터 분리
from sklearn.model_selection import train_test_split
xt,xv,yt,yv = train_test_split(
    train,
    target,
    test_size=0.2,
    random_state=42
)

#모델 학습 및 평가(회귀)
import numpy as np
#평가 모듈 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#학습 모듈
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
lr.fit(xt,yt)
pred = lr.predict(xv) # 학습한 회귀식으로 xv 예측

print(r2_score(yv,pred))
print(mean_absolute_error(yv,pred))
print(mean_squared_error(yv,pred))

#평가 및 제출
pred = lr.predict(test)
pd.DataFrame({'id' : test_id, 'output' : pred}).to_csv("19194.csv", index = False)
print(pd.read_csv('19194.csv'))


