import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 讀取CSV文件
df=pd.read_csv(r'D:\Users\zi tong\Desktop\code\DL\0423_hw\supermarket_sales - Sheet1.csv')

print(df)

# 處理數據，選擇特徵和目標變量
X = df[['Unit price','Quantity']]  # 選擇你的特徵列
y = df[ 'gross income']  # 選擇你的目標變量列

# 將數據集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

# 特徵標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化ElasticNet模型，並設置L1和L2正則化參數
elastic_net = ElasticNet(alpha=0.35, l1_ratio=0.5)  # alpha為正則化強度，l1_ratio為L1正則化的比例

# 在訓練集上訓練模型
elastic_net.fit(X_train, y_train)

# 在測試集上進行預測
predictions = elastic_net.predict(X_test)


# 繪製scatter plot
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predicted Values')
plt.show()

# 計算準確率
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
