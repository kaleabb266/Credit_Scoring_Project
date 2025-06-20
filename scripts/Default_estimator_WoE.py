import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Calculate normalized RFMS features and composite score
def calculate_rfms(df):
    # Ensure TransactionStartTime is datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Recency'] = (df['TransactionStartTime'].max() - df['TransactionStartTime']).dt.days
    df['Frequency'] = df['TransactionCount']
    df['Monetary'] = df['TotalTransactionAmount']
    df['Seasonality'] = df['TransactionMonth']

    # Normalize each component to [0, 1]
    for col in ['Recency', 'Frequency', 'Monetary', 'Seasonality']:
        df[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Composite RFMS score (sum of normalized features)
    df['RFMS_Score'] = df[['Recency_norm', 'Frequency_norm', 'Monetary_norm', 'Seasonality_norm']].sum(axis=1)
    return df

# Visualize RFMS space (pairwise scatter plots)
def visualize_rfms(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Recency_norm'], df['Monetary_norm'], c='blue', label='Recency vs Monetary')
    plt.scatter(df['Frequency_norm'], df['Seasonality_norm'], c='red', label='Frequency vs Seasonality')
    plt.xlabel('Normalized RFMS Components')
    plt.ylabel('Normalized Values')
    plt.title('RFMS Space Visualization')
    plt.legend()
    plt.show()

# Create a proxy for default estimation with robust label assignment
def create_default_estimator(df):
    features = ['Recency_norm', 'Frequency_norm', 'Monetary_norm', 'Seasonality_norm']
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['RFMS_Cluster'] = kmeans.fit_predict(df[features])

    # Assign "Good" to cluster with higher mean RFMS_Score
    cluster_means = df.groupby('RFMS_Cluster')['RFMS_Score'].mean()
    good_cluster = cluster_means.idxmax()
    df['Label'] = np.where(df['RFMS_Cluster'] == good_cluster, 'Good', 'Bad')
    return df

# WoE binning function with smoothing
def woe_binning(df, target_col, feature):
    # Bin the continuous feature into 10 quantile-based bins
    df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')

    # Calculate total number of "Good" and "Bad"
    total_good = (df[target_col] == 'Good').sum()
    total_bad = (df[target_col] == 'Bad').sum()

    # Calculate WoE for each bin
    woe_values = {}
    grouped = df.groupby('bin')
    for name, group in grouped:
        good_count = (group[target_col] == 'Good').sum()
        bad_count = (group[target_col] == 'Bad').sum()
        good_prop = good_count / total_good if total_good > 0 else 0
        bad_prop = bad_count / total_bad if total_bad > 0 else 0
        # Add smoothing to avoid division by zero
        woe_values[name] = np.log((good_prop + 1e-6) / (bad_prop + 1e-6))

    # Map WoE values back to the DataFrame
    df['WoE'] = df['bin'].map(woe_values)
    return df
