import pandas as pd

# age 컬럼의 이상치를 제거 (소수점 나이와 음수나이, 0포함)
# 제거 전 후의 views 컬럼 표준편차를 더하시오 (최종 결과 값은 소수 둘째자리까지 출력, 셋째자리에서 반올림)

df = pd.read_csv('members.csv')

#나이가 0과 음수인 행을 제거.

#print(df.shape)
positive_age = df['age'] > 0

#소수점 나이 제거
integer_age = (df['age'] - df['age'].astype(int))==0

new_df = df[positive_age & integer_age]
#print(new_df.shape)
print(round(df['views'].std() + new_df['views'].std(),2))