
# 에어비엔비 가격?
# 평가: R-Squared, MAE, MSE, RMSE, RMSLE, MAPE
# target : price(가격) -> 회귀
# csv파일 생성 : 수험번호.csv (예시 아래 참조)

import pandas as pd

train = pd.read_csv('content/practice3/train.csv')
test = pd.read_csv('content/practice3/test.csv')

#총 컬럼수 확인.
print(train.shape, test.shape)

#데이터 타입 확인 -> object가 혼재함으로 인코딩 필요.
#baseline으로 drop을 할지, 인코딩을 진행할지 선택 필요.
print(train.info(), test.info())

#결측값 확인 -> 결측값이 있는 컬럼 범위 동일
print(train.isnull().sum())
print(test.isnull().sum())

#train 데이터에서 target 값 분리.
target = train['price']

#target 데이터의 범위 분석
print(target.value_counts())
print(target.describe())

#이상치 데이터 확인, price 인데 0원인 것은 이상값으로 확인하고 제거.
#만약에 price 0인 것을 제외하려면, test set이랑 갯수가 동일해야함으로
#union 한 후에 분리해줘야 할 것으로 보임
#이상값을 제거한 다음에 pop 해야함.

## 2. 데이터 전처리
print(train.nunique())
#nunique를 찍어서 결측치가 존재하던 컬럼들 확인
#host_name, name은 삭제하고, last_review도 삭제
#reviews per month는 0으로 결측치 처리, host_id도 과적합이 있을 수 있어서 삭제

cols = ['name','host_name','last_review','host_id']
print(train.shape)
train = train.drop(cols,axis=1)
test = test.drop(cols,axis=1)
print(train.shape)

train['reviews_per_month'] = train['reviews_per_month'].fillna(0)
test['reviews_per_month'] = test['reviews_per_month'].fillna(0)

print(train.isnull().sum())
train = train.drop('id',axis=1)
test_id = test.pop('id')

print(test.head())

print(train.info())
#train 데이터에서 object들을 레이블 인코딩 진행

cols = train.select_dtypes(include='object').columns
print(cols)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cols :
    train[col] = le.fit_transform(train[col]) #train은 실제 인코딩 변환값을 적용함으로 fit_transform
    test[col] = le.transform(test[col]) #test는 실제 인코딩 변환값을 적용해서는 안됨으로 fit 사용

print(train.shape, test.shape)
print(train.info())

#모델 분할
from sklearn.model_selection import train_test_split

xt,xv,yt,yv = train_test_split(
    train.drop('price',axis=1),
    target,
    test_size=0.2,
    random_state=0
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xt,yt) #모델 생성 xt는 사례, yt는 정답
pred = model.predict(xv) #xv를 예측

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

print(r2_score(yv,pred))
print(root_mean_squared_error(yv,pred))
print(mean_absolute_error(yv,pred))

pred = model.predict(test)
submit = pd.DataFrame({
    'id' : test_id,
    'pred': pred
})

submit.to_csv('prc2_2026',index=False)

file = pd.read_csv('prc2_2026')
print(file.head())

#xt 정답을 제외한 피처 데이터 -> 훈련에 사용
#xv 모의고사 문제 predict(xv)로 실제 정답지 정답확인훈련 (yv)
#yt xt와 완벽히 일치하는 정답데이터 -> 훈련에 사용
#yv 모의고사 정답지 모의고사(xv)를 바탕으로 출력한 정답(yv)