import pandas as pd
import matplotlib.pyplot as plt

# Sample data (replace with your actual data loading if needed)
data = {
    'Date': ['2019-03-11', '2019-03-12', '2019-03-13', '2019-03-14', '2019-03-15'],
    'Open': [1.08, 1.08, 1.06, 1.06, 1.06],
    'High': [1.10, 1.09, 1.08, 1.11, 1.08],
    'Low': [1.08, 1.05, 1.04, 1.06, 1.04],
    'Close': [1.08, 1.05, 1.07, 1.08, 1.04],
    'Adj Close': [1.08, 1.05, 1.07, 1.08, 1.04],
    'Volume': [32100, 20200, 23100, 29900, 30900]
}
df = pd.DataFrame(data)

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(df['Date'], df['Open'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Open')
plt.title('Open vs. Date (First 5 Rows)')
plt.grid(True)
plt.savefig('plots/image.png')
