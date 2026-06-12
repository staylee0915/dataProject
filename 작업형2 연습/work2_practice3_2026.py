import pandas as pd

#성별, 나이, 혈압, 콜레스테롤, 공복혈당, 최대심박수
# roc-auc, 정확도(accuracy), f1을 구하시오
#target = output (1 심장마비 높음 - 0 낮음)
# csv 파일생성
# id, output

train = pd.read_csv('content/practice4/train.csv')
test = pd.read_csv('content/practice4/test.csv')

print(train.shape, test.shape)

#결측값 없고, object없어서 인코딩 필요 없음.
print(train.info())
print(train.isnull().sum())

target = train.pop('output')
print(target.describe())
#0,1 분류분석, 균등분포.
print(target.value_counts())

#과적합을 막기 위해 id 분리
print(train.value_counts())
print(test.value_counts())
train = train.drop('id',axis = 1)
test_id = test.pop('id')

#모델은 분류분석으로 진행
from sklearn.model_selection import train_test_split

xt,xv,yt,yv = train_test_split(
    train,
    target,
    test_size = 0.2,
    random_state = 2022
)

#학습 모델 선정
#
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀는 분류분석임

logi = LogisticRegression()
logi.fit(xt,yt)
pred = logi.predict(xv)

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score

print(accuracy_score(yv,pred))
print(f1_score(yv,pred))

pred_proba = logi.predict_proba(xv)
print(roc_auc_score(yv,pred_proba[:,1]))

# 0.8163265306122449
# 0.8524590163934426
# 0.8982758620689656

pred = logi.predict(test)

result = pd.DataFrame({
    'id' : test_id,
    'pred' : pred
})

result.to_csv('2026-3.csv',index = False)
print(pd.read_csv('2026-3.csv'))