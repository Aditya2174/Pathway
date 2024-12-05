import pandas as pd
import matplotlib.pyplot as plt

data = '''Date,Open,High,Low,Close,Adj Close,Volume
2019-03-11,1.08,1.10,1.08,1.08,1.08,32100
2019-03-12,1.08,1.09,1.05,1.05,1.05,20200
2019-03-13,1.06,1.08,1.04,1.07,1.07,23100
2019-03-14,1.06,1.11,1.06,1.08,1.08,29900
2019-03-15,1.06,1.08,1.04,1.04,1.04,30900'''

df = pd.read_csv(pd.compat.StringIO(data))
df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(10, 6))
plt.plot(df['Date'][:5], df['Open'][:5], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Open')
plt.title('Open vs. Date (First 5 Rows)')
plt.grid(True)
plt.savefig('plots/image.png')
