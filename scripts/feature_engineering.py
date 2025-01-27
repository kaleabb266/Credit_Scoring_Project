import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def total_transaction_amount(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the total transaction amount per customer.
    """
    return df.groupby('CustomerId')['Amount'].sum()

# Calculate average transaction amount per customer
def average_transaction_amount(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the average transaction amount per customer.
    """
    return df.groupby('CustomerId')['Amount'].mean()

# Calculate transaction count per customer
def transaction_count(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the count of transactions per customer.
    """
    return df.groupby('CustomerId')['TransactionId'].count()

# Calculate standard deviation of transaction amounts per customer
def transaction_std_dev(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the standard deviation of transaction amounts per customer.
    """
    return df.groupby('CustomerId')['Amount'].std()

# Main function to compute all metrics and return a consolidated DataFrame
def compute_customer_metrics(df):
    """
    Compute the total, average, count, and standard deviation of transactions for each customer.
    """
    

    # Calculate each metric
    total_amount = total_transaction_amount(df)
    avg_amount = average_transaction_amount(df)
    count_transactions = transaction_count(df)
    std_dev_amount = transaction_std_dev(df)

    # Consolidate the results into a DataFrame
    result = pd.DataFrame({
        'TotalTransactionAmount': total_amount,
        'AverageTransactionAmount': avg_amount,
        'TransactionCount': count_transactions,
        'TransactionStdDev': std_dev_amount
    })

    return result



# Function to convert transaction start time to datetime and extract features
def extract_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts transaction hour, day, month, and year from the 'TransactionStartTime' field.
    Assumes 'TransactionStartTime' is in a datetime-compatible format.
    """
    # Convert 'TransactionStartTime' to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extract the desired features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year

    return df[['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']]





# Function to perform One-Hot Encoding on a single column
def one_hot_encode_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Apply one-hot encoding to a specific column in the DataFrame.
    """
    one_hot_encoded_df = pd.get_dummies(df[column], prefix=column)
    df = df.drop(column, axis=1)  # Drop the original column
    df = pd.concat([df, one_hot_encoded_df], axis=1)  # Concatenate the encoded columns
    return df

# Function to perform Label Encoding on a single column
def label_encode_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Apply label encoding to a specific column in the DataFrame.
    """
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df

# Example usage function
def encode_categorical_columns(df: pd.DataFrame, columns: list, encoding_type: str = 'onehot') -> pd.DataFrame:
    """
    Encodes the specified categorical columns using either one-hot or label encoding.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to be encoded.
        encoding_type (str): 'onehot' for one-hot encoding, 'label' for label encoding.
    """
    for column in columns:
        if encoding_type == 'onehot':
            df = one_hot_encode_column(df, column)
        elif encoding_type == 'label':
            df = label_encode_column(df, column)
        else:
            raise ValueError("Unsupported encoding type. Use 'onehot' or 'label'.")
    
    return df




# Function to normalize a single column (Min-Max Scaling)
def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Apply normalization (scaling values between 0 and 1) to a specific column in the DataFrame.
    """
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])  # Normalize the column
    return df

# Function to standardize a single column (Z-Score Scaling)
def standardize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Apply standardization (mean 0, std dev 1) to a specific column in the DataFrame.
    """
    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[[column]])  # Standardize the column
    return df

# Function to apply either normalization or standardization to multiple columns
def scale_numerical_columns(df: pd.DataFrame, columns: list, scaling_type: str = 'normalize') -> pd.DataFrame:
    """
    Scales the specified numerical columns using either normalization or standardization.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to be scaled.
        scaling_type (str): 'normalize' for normalization, 'standardize' for standardization.
    """
    for column in columns:
        if scaling_type == 'normalize':
            df = normalize_column(df, column)
        elif scaling_type == 'standardize':
            df = standardize_column(df, column)
        else:
            raise ValueError("Unsupported scaling type. Use 'normalize' or 'standardize'.")
    
    return df




