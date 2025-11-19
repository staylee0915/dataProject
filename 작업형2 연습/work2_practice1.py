# 제공된 데이터는 10개 아울렛 매장에서 1500여개 제품의 판매 데이터를 수집한 것.
# 학습용 데이터를 기반으로 판매금액을 예측하는 모델 개발
# 개발한 모델을 평가용 데이터에 적용하여 예측겨로가를 다음 제출 형식에 따라 result.csv 파일로 생성

# 평가지교 RMSE
# 예측ㄷ상 : Item_Outlet_Sales
# 제출하일 : 예측값만 포함, 컬럼명은 pred로 지정

import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('content/practice1/train.csv')
test = pd.read_csv('content/practice1/test.csv')


#데이터 크기
print(train.shape)
print(test.shape)

#데이터 타입
print(train.info())
print(test.info())

#수치형 컬럼값 확인
print(train.describe())
target = train['Item_Outlet_Sales']

#범주형 통계값 확인
print(train.describe(include='object'))
#판매금액 예측임으로, 회귀문제 랜덤포레스트회귀 함수 사용
#train과 test로 분할해야함.

print(train.isnull().sum())
print(test.isnull().sum())

### 데이터 전처리 (타겟 분리)

target = train.pop('Item_Outlet_Sales')
#결측치 처리 최소, 중앙, 평균 중에 선택

train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].median())
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
#test에 train최빈값으로 채움.
test['Item_Weight'] = test['Item_Weight'].fillna(train['Item_Weight'].median())
test['Outlet_Size'] = test['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])

#인코딩 진행 (문자형 -> 숫자형으로 변형해 주는 것.)
cols = list(train.columns[train.dtypes == object])
print(train.shape,test.shape)
df = pd.concat([train,test])
print(df.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cols :
    df[col] = le.fit_transform(df[col])

print(df.head())

train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):].copy()

#검증 데이터 분할
from sklearn.model_selection import train_test_split

xt, xv, yt, yv = train_test_split(
    train,
    target,
    test_size = 0.2,
    random_state = 0
)
print(xt.shape, xv.shape, yt.shape, yv.shape)

#검증
from sklearn.metrics import root_mean_squared_error

#선형회귀로 모델 제작
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#fit을 통해 학습 진행
lr.fit(xt, yt)
#예측진행
y_pred = lr.predict(xv)

#실제 target의 validation 값과 y의 예측값을 통해 검증
result = root_mean_squared_error(yv,y_pred)

print(result)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(xt,yt)
y_pred = rf.predict(xv)

result = root_mean_squared_error(yv,y_pred)
print(result)

pred = rf.predict(test)
print(pred)

submit = pd.DataFrame({'pred':pred})
submit.to_csv("result.csv",index = False)

print(pd.read_csv('result.csv'))