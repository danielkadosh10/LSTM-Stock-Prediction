import pandas as pd
from sklearn.preprocessing import StandardScaler


def data_processing():
    data = 'data.csv'
    df = pd.read_csv(data)[7500:8500]  # Sample 1000 different days in the dataset

    df.describe()

    autoscaler = StandardScaler()
    X = df[["Open", "High", "Low", "Adj Close", "Volume"]]  # Features data
    Y = df['Close']  # Target label
    split_index = int(0.7 * len(df))  # 70/30 training test split

    X_train = autoscaler.fit_transform(X[:split_index])
    X_test = autoscaler.transform(X[split_index:])
    Y_train = autoscaler.fit_transform(Y[:split_index].values.reshape(-1, 1)).flatten()  # Transform Y_train
    Y_test = autoscaler.transform(Y[split_index:].values.reshape(-1, 1)).flatten()  # Transform Y_test

    return X_train, X_test, Y_train, Y_test, autoscaler
