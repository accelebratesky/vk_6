
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn.preprocessing import StandardScaler


def build_dataset(csv_path="./data/sine.csv", window_size=20, train_ratio=0.8):
    # tiqudata
    data = pd.read_csv(csv_path)
    data_x = data["value"].to_numpy(dtype=float)
    print(data_x.shape)

    # 接下来干嘛，划分数据集吧，注意不能用之前的那种随机划分,直接原始切割吧
    percent = train_ratio
    train_num = int(len(data_x) * percent)
    train_data_x = data_x[:train_num]
    test_data_x = data_x[train_num:]
    print(train_data_x.shape)
    print(test_data_x.shape)

    # 归一化一下吧
    train_data_x = train_data_x.reshape(-1, 1)
    test_data_x = test_data_x.reshape(-1, 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data_x)
    X_test_scaled = scaler.transform(test_data_x)

    #接下来写个切窗吧
    def cutcut(data, size):
        X, y = [], []
        xunhuan = len(data) - size
        for i in range(xunhuan):
            X.append(data[i:i + size])
            y.append(data[i + size])
        return np.array(X), np.array(y)


    X_train, y_train = cutcut(X_train_scaled, window_size)


    combined = np.concatenate([X_train_scaled, X_test_scaled])
    split = train_num
    X_test, y_test = [], []
    for i in range(split, len(combined)):
        X_test.append(combined[i - window_size:i])
        y_test.append(combined[i])
    X_test = np.array(X_test)
    y_test = np.array(y_test)


    X_train = Tensor(X_train)
    X_test = Tensor(X_test)
    y_train = Tensor(y_train)
    y_test = Tensor(y_test)

    return X_train, y_train, X_test, y_test, scaler
