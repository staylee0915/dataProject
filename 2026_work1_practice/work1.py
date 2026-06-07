import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/lovedlim/inf/refs/heads/main/p1/type1_data1.csv")

df = df.dropna()

df['new'] = df['views'] / df['f1']
print(df.head())

answer = df.sort_values('new',ascending = False)

print(int(answer['age'].iloc[0]))
