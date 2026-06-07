'''
나이, 급여, 결혼상태, 신용카드한도, 신용카드 카테고리등의 컬럼
평가 roc-auc(분류), 정확도, f1, 정밀도, 재현율을 구하시오

target : attirtion_flag 1: 이탈, 0 유지
csv 파일 생성 : 수험번호.csv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

각각의 함수에 (y_test, pred)를 넣어서 동작.
'''


import pandas as pd

train = pd.read_csv('content/practice2/train.csv')
test = pd.read_csv('content/practice2/test.csv')

#train, test 데이터의 분포 확인
print(train.info())
print(test.info())

#null 값은 없음을 확인 가능.
print(train.isnull().sum())
print(test.isnull().sum())

#각 속성의 갯수 확인.
print(train.shape)
print(test.shape)

#value counts로 attrition_flag의 값의 범위 확인.
#불균일 데이터임을 확인 할 수 있음.
target = train['Attrition_Flag']
print(target.value_counts())

cols = train.select_dtypes(include='object').columns
print("cols", cols)
#과적합을 방지하기 위해 client num 제외 train과 test 모두 제외해야함.
#단 test의 경우 제출할 시에 id 값이 붙어야 함으로 포함함.
train = train.drop('CLIENTNUM',axis = 1)
train = train.drop(cols,axis=1)
train = train.drop('Attrition_Flag',axis=1)
test_id = test.pop('CLIENTNUM')
print(train.shape, test.shape)

print(test.head(1))

#학습시작
#1. 검증 데이터 분리

from sklearn.model_selection import train_test_split

xt, xv, yt, yv = train_test_split(
    train,
    target, # train value
    test_size=0.2,
    random_state=2022
)

#2. 모델 학습
#학습, 예측

from sklearn.ensemble import RandomForestClassifier #분류임으로 classifier로

model = RandomForestClassifier()
model.fit(xt, yt)
pred = model.predict(xv)

#평가
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print(accuracy_score(yv,pred))
print(recall_score(yv,pred))
print(f1_score(yv,pred))
print(precision_score(yv,pred))

#base line
# 0.9666872301048736
# 0.851145038167939
# 0.892
# 0.9369747899159664

## roc-auc는 predict proba로 
pred = model.predict_proba(xv)
print(pred)
print(roc_auc_score(yv,pred[:,1])) #양성에 해당하는 값을 반환받기 위해서

# 0.9905492925309922


#label incoding

#전처리
train = pd.read_csv('content/practice2/train.csv')
test = pd.read_csv('content/practice2/test.csv')

#타겟 분리
target = train.pop('Attrition_Flag')

#과적합 처리
train = train.drop('CLIENTNUM', axis=1)
test_id = test.pop('CLIENTNUM')

#인코딩할 컬럼들 분리
cols = train.select_dtypes(include='object').columns

#인코딩은 범주형 데이터를 숫자형태로 변환하는 과정
from sklearn.preprocessing import LabelEncoder

#라벨 인코딩은 한번씩만, 따라서 반복문으로 사용
for col in cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

print(train[cols].head())

from sklearn.model_selection import train_test_split

xt,xv,yt,yv = train_test_split(
    train,
    target,
    test_size=0.2,
    random_state=2022
)

#모델 학습

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(xt,yt)
#실제 제출할때는 test를 예측해야한다.
#현재는 xt로 yt를 예측해서 전체 갯수가 동일하지만,
#나중에는 모델로 test를 예측해서 (xv대신 test) 그 결과를 csv로 만들어야한다.
#따라서 xt, yt로 xv로 pred값을 만들고, 이를 yv와 비교해서 평가하지만
#결과적으로 제출해야하는 것은 test 데이터이며, 그것은 평가값을 알 수 없다.
pred = rf.predict(xv)

from sklearn.metrics import f1_score
print(f1_score(yv,pred))
# f1_score label 인코딩결과
#0.8866

#one-hot 인코딩

#전처리
train = pd.read_csv('content/practice2/train.csv')
test = pd.read_csv('content/practice2/test.csv')

#타겟 분리
target = train.pop('Attrition_Flag')

#과적합 처리
train = train.drop('CLIENTNUM', axis=1)
test_id = test.pop('CLIENTNUM')

train = pd.get_dummies(train,columns = cols)
test = pd.get_dummies(test,columns=cols)

from sklearn.model_selection import train_test_split

xt,xv,yt,yv = train_test_split(
    train,
    target,
    test_size=0.2,
    random_state=2022
)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(xt,yt)
#실제 제출할때는 test를 예측해야한다.
#현재는 xt로 yt를 예측해서 전체 갯수가 동일하지만,
#나중에는 모델로 test를 예측해서 (xv대신 test) 그 결과를 csv로 만들어야한다.
#따라서 xt, yt로 xv로 pred값을 만들고, 이를 yv와 비교해서 평가하지만
#결과적으로 제출해야하는 것은 test 데이터이며, 그것은 평가값을 알 수 없다.
pred = rf.predict(xv)

from sklearn.metrics import f1_score
print(f1_score(yv,pred))

# 원핫 인코딩 결과
# 0.8752
#predict는 분류
#predict proba는 확률
pred = rf.predict_proba(test)

submit = pd.DataFrame(
    {
        'CLIENTNUM':test_id,
        'Attrition_Flag':pred[:,1]
    }
)

print(submit)
submit.to_csv('2026-2.csv', index = False)

y_test = pd.read_csv('content/practice2/y_test.csv')
print(roc_auc_score(y_test,pred[:,1]))