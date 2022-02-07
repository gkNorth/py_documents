import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('bike.tsv', sep='\t')
# print(df)
weather = pd.read_csv('weather.csv', encoding='shift-jis')
# print(weather)
temp = pd.read_json('temp.json').T
# print(temp)
df2 = df.merge(weather, how='inner', on='weather_id')
# print(df2.columns)
# print(df2.groupby('weather').mean()['cnt'])

# print(temp.loc[199:201])
# print(df2[df2['dteday']=='2011-07-20'])
df3 = df2.merge(temp, how='left', on='dteday')
# print(df3)

# df3[['temp', 'hum']].plot(kind='line')
# df3['temp'].plot(kind='hist')
# df3['hum'].plot(kind='hist', alpha=0.5)

df3['atemp'] = df3['atemp'].astype(float)
df3['atemp'] = df3['atemp'].interpolate()

df3['atemp'].loc[220:240].plot(kind='line')
plt.show()

# print(df3['atemp'])