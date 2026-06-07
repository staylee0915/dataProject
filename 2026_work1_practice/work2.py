import pandas as pd
# df = pd.read_csv("type1_data1.csv")
df = pd.read_csv("https://raw.githubusercontent.com/lovedlim/inf/refs/heads/main/p1/type1_data1.csv")


df = df.fillna(0)

df = df.sort_values('views',ascending = False)

minvalue = df['views'].iloc[9] #10번째 값을 가져온다.

df.loc[df.index[:10], 'views'] = minvalue

print(int(df['views'].sum()))
#df['views'][:10].replace(df['views'].iloc(10))
