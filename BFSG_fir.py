import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt

# 讀取CSV文件
df=pd.read_csv(r'D:\Users\zi tong\Desktop\code\DL\0423_hw\supermarket_sales - Sheet1.csv')

print(df)

# 處理數據，選擇特徵和目標變量
X = df[['Unit price','Quantity']]  # 選擇你的特徵列
y = df[ 'gross income']  # 選擇你的目標變量列

# 定義模型，加入L2正則化
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

# 編譯模型，使用Adam優化算法和指定的學習率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 定義模型，加入L2正則化
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

# 編譯模型，使用Adam優化算法和指定的學習率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 訓練模型
history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)

# 繪製loss曲線
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


