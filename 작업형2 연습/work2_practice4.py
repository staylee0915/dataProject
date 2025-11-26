import pandas as pd

#성별, 나이, 혈압, 콜레스테롤, 공복혈당, 최대심박수
# roc-auc, 정확도(accuracy), f1을 구하시오
#target = output (1 심장마비 높음 - 0 낮음)
# csv 파일생성
# id, output

train = pd.read_csv('content/practice4/train.csv')
test = pd.read_csv('content/practice4/test.csv')

print(train.shape)
print(test.shape)

print(train.info())
print(test.info())

print(train.isnull().sum())
print(test.isnull().sum())

print(train.describe())

#타겟 데이터 분리
target = train.pop('output')

#결측값 없음.
#object없음, 인코딩 불필요, 과적합 막기 위해 id 삭제
print(train.head(3))
print(train.nunique())
train = train.drop('id',axis = 1)
test_id = test.pop('id')
print(train.shape, test.shape)
# 데이터셋 분리
from sklearn.model_selection import train_test_split

xt,xv,yt,yv = train_test_split(
    train,
    target,
    test_size = 0.2,
    random_state = 42
)

#학습 분류임으로 랜덤포레스트 사용
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42, max_depth = 5, n_estimators=200)
model.fit(xt,yt)
pred = model.predict(xv)
print(pred)



from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
pred_proba = model.predict_proba(xv)
print(roc_auc_score(yv,pred_proba[:,1]))
print(f1_score(yv,pred))
print(accuracy_score(yv,pred))

#xgboast 사용
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 42, max_depth = 5, n_estimators = 600, learning_rate=0.01)
xgb.fit(xt,yt)
pred = xgb.predict(xv)
pred_proba = xgb.predict_proba(xv)

print(roc_auc_score(yv,pred_proba[:,1]))
print(f1_score(yv,pred))
print(accuracy_score(yv,pred))

pred = model.predict(test)
pd.DataFrame({'id' : test_id,'output' : pred}).to_csv('content/practice4/249194.csv',index=False)
print(pd.read_csv('content/practice4/249194.csv'))
