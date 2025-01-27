import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Assume data is in a DataFrame called `df`
# Define the RFMS features based on the columns provided
def calculate_rfms(df):
    df['Recency'] = (df['TransactionStartTime'].max() - df['TransactionStartTime']).dt.days
    df['Frequency'] = df['TransactionCount']
    df['Monetary'] = df['TotalTransactionAmount']
    df['Seasonality'] = df['TransactionMonth']
    df['RFMS_Score'] = (df['Recency'] + df['Frequency'] + df['Monetary'] + df['Seasonality'])
    return df

# Visualize RFMS space
def visualize_rfms(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Recency'], df['Monetary'], c='blue', label='Recency vs Monetary')
    plt.scatter(df['Frequency'], df['Seasonality'], c='red', label='Frequency vs Seasonality')
    plt.xlabel('RFMS Components')
    plt.ylabel('Values')
    plt.title('RFMS Space Visualization')
    plt.legend()
    plt.show()

# Create a proxy for default estimation
def create_default_estimator(df):
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['RFMS_Cluster'] = kmeans.fit_predict(df[['Recency', 'Frequency', 'Monetary', 'Seasonality']])
    df['Label'] = np.where(df['RFMS_Cluster'] == 0, 'Good', 'Bad')
    return df

# Fixed WoE binning function
def woe_binning(df, target_col, feature):
    # Bin the continuous feature into 10 discrete categories
    df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
    
    # Group by the bins
    grouped = df.groupby('bin')
    
    # Calculate total number of "good" and "bad" labels
    total_good = (df[target_col] == 'Good').sum()
    total_bad = (df[target_col] == 'Bad').sum()
    
    # Calculate WoE for each bin
    woe_values = {}
    for name, group in grouped:
        good_count = (group[target_col] == 'Good').sum()
        bad_count = (group[target_col] == 'Bad').sum()
        
        good_prop = good_count / total_good if total_good > 0 else 0
        bad_prop = bad_count / total_bad if total_bad > 0 else 0
        
        woe_values[name] = np.log((good_prop + 1e-6) / (bad_prop + 1e-6))
    
    # Map the WoE values back to the original rows based on their bin
    df['WoE'] = df['bin'].map(woe_values)
    return df
