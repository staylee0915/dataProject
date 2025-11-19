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

#탐색적 데이터 분석
print(train.shape)
print(test.shape)

print(train.describe)
print(test.describe)

print(train.info())
print(test.info())

#attrition_flag의 value counts를 확인
print(train['Attrition_Flag'].value_counts())

print(train.isnull().sum())
print(test.isnull().sum())

#######
##데이터 전처리 & 피처 엔지니어링

## baseline 모델 생성
## object 타입의 컬럼만 선택
cols = train.select_dtypes(include='object').columns

#object 컬럼을 drop (인코딩 없이 진행)
print(train.shape,test.shape)
train = train.drop(cols,axis=1)
train = train.drop('CLIENTNUM',axis=1) ## 오버피팅 나지 않게 
target = train['Attrition_Flag']
test_id = test.pop('CLIENTNUM')
test = test.drop(cols,axis=1)
print(train.shape, test.shape)

#검증데이터 분리
from sklearn.model_selection import train_test_split

xt,xv,yt,yv = train_test_split(
    train.drop('Attrition_Flag',axis = 1),
    target,
    test_size = 0.2,
    random_state = 2022
)

## 원핫 인코딩
## 레이블 인코딩


####
## 모델 학습
# 모델불러오기
# 학습
# 예측

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(xt,yt) # 학습
pred = rf.predict(xv) # 예측
print(pred)

#예측값 평가
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, roc_auc_score

print(accuracy_score(yv,pred),precision_score(yv,pred),recall_score(yv,pred),f1_score(yv,pred))
#base line
#0.9648365206662554 0.9399141630901288 0.8358778625954199 0.8848484848484849

#예측값 평가 ruc auc
pred = rf.predict_proba(xv)
print(roc_auc_score(yv,pred[:,1]))
#baseline
#0.9904032489088856