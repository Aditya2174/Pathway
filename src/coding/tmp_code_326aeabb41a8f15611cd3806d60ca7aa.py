import matplotlib.pyplot as plt
import pandas as pd
import os

# Create the plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

data = """05  1.24 1.05   1.14       1.14    87100
2020-02-26  1.18  1.18 1.05   1.13       1.13    24600
2020-02-27  1.13  1.13 1.05   1.06       1.06    35300
2020-02-28  1.02  1.07 0.95   1.03       1.03    45900
2020-03-02  1.00  1.00 0.98   0.98       0.98    13500"""

# Create a DataFrame from the data
df = pd.read_csv(pd.compat.StringIO(data), delim_whitespace=True, header=None)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Plot the first five rows
plt.figure(figsize=(10, 6))
plt.plot(df['Date'][:5], df['Open'][:5], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Open')
plt.title('Open vs. Date (First 5 Rows)')
plt.grid(True)
plt.savefig('plots/image.png')
