import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

# 기본 설정
data_dir = './miniData'
dates = [f'202202{day:02d}' for day in range(1, 29)]
hours = [f'{h:02d}' for h in range(24)]

# shape 확인
# 데이터 성별 설정
gender = 'm' # 'm' 또는 'w'
sample_path = os.path.join(data_dir, f'{dates[0]}_{gender}_2529_P_T_00.csv')
sample = np.loadtxt(sample_path, delimiter=',')
height, width = sample.shape  # 예: 72, 49

# 중앙 5x5 영역 인덱스
center_h = height // 2
center_w = width // 2
h_start = center_h - 2
h_end = center_h + 3
w_start = center_w - 2
w_end = center_w + 3

# 학습용 데이터 (1~27일)
X_train = np.zeros((27, 24, 5, 5))
for d_idx, date in enumerate(dates[:27]):
    for h_idx, hour in enumerate(hours):
        file_path = os.path.join(data_dir, f'{date}_{gender}_2529_P_T_{hour}.csv')
        if os.path.exists(file_path):
            full_img = np.loadtxt(file_path, delimiter=',')
            X_train[d_idx, h_idx] = full_img[h_start:h_end, w_start:w_end]
        else:
            print(f'누락 파일 이릉: {file_path}')

# 테스트용 데이터 (28일)
y_test = np.zeros((24, 5, 5))
for h_idx, hour in enumerate(hours):
    file_path = os.path.join(data_dir, f'20220228_{gender}_2529_P_T_{hour}.csv')
    if os.path.exists(file_path):
        full_img = np.loadtxt(file_path, delimiter=',')
        y_test[h_idx] = full_img[h_start:h_end, w_start:w_end]
    else:
        print(f'누락 파일 이름: {file_path}')

print('X_train shape:', X_train.shape)  # (27, 24, 5, 5)
print('y_test shape:', y_test.shape)    # (24, 5, 5)

# 슬라이딩 윈도우 생성
window_size = 3
X = []
y = []

for day in range(X_train.shape[0]):
    for t in range(24 - window_size):
        input_window = X_train[day, t:t+window_size]  # (3, 5, 5)
        target = X_train[day, t+window_size]          # (5, 5)
        X.append(input_window)
        y.append(target)

X = np.array(X)  # (samples, 3, 5, 5)
y = np.array(y)  # (samples, 5, 5)

# 입력 형태 변경: (samples, 5, 5, 3)
X = X.transpose(0, 2, 3, 1)
y = y[..., np.newaxis]

# 정규화 (0~1 범위)
X_min, X_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()

X = (X - X_min) / (X_max - X_min)
y = (y - y_min) / (y_max - y_min)

# 학습/검증 분리
split_idx = int(len(X) * 0.8)
X_train_dense = X[:split_idx]
y_train_dense = y[:split_idx]
X_val_dense = X[split_idx:]
y_val_dense = y[split_idx:]

# Dense 모델 정의
model = Sequential([
    Input(shape=(5, 5, 3)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(25, activation='linear'),
    Reshape((5, 5, 1))
])

model.compile(optimizer=Adam(), loss='mse')

# 모델 학습
history = model.fit(X_train_dense, y_train_dense, validation_data=(X_val_dense, y_val_dense),
                    epochs=20, batch_size=16)

# 28일 순차 예측 (27일 21~23시 기반)
X_forecast = X_train[-1, 21:24]  # 27일 마지막 날 21~23시 (3, 5, 5)
X_forecast = X_forecast.transpose(1, 2, 0)  # (5, 5, 3)
X_forecast = (X_forecast - X_min) / (X_max - X_min)  # 정규화

predicted_28 = []

for h in range(24):
    input_tensor = X_forecast[np.newaxis, ...]  # (1, 5, 5, 3)
    pred = model.predict(input_tensor)[0, ..., 0]  # 예측 (5, 5)
    predicted_28.append(pred)

    # 새 입력 구성 (이전 2시간 + 예측값)
    X_forecast = np.concatenate([X_forecast[..., 1:], pred[..., np.newaxis]], axis=-1)

predicted_28 = np.array(predicted_28)  # shape: (24, 5, 5)

# y_test 정규화
y_test_scaled = (y_test - y_min) / (y_max - y_min)

# 전체 MSE
from sklearn.metrics import mean_squared_error
mse_28 = mean_squared_error(y_test_scaled.reshape(-1), predicted_28.reshape(-1))
print(f' 28일 전체 예측 MSE: {mse_28:.6f}')


fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(18, 12))
fig.suptitle('Actual (28day)', fontsize=16)

for i in range(24):
    row, col = divmod(i, 6)
    
    # 실제값
    ax_actual = axes[row][col]
    ax_actual.imshow(y_test_scaled[i], cmap='Reds')
    ax_actual.set_title(f'hour: {i:02d}', fontsize=10)
    ax_actual.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 예측값은 아래쪽에 따로 한번에
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(18, 12))
fig.suptitle('Predicted (28day)', fontsize=16)

for i in range(24):
    row, col = divmod(i, 6)
    
    ax_pred = axes[row][col]
    ax_pred.imshow(predicted_28[i], cmap='Reds')
    ax_pred.set_title(f'hour: {i:02d}', fontsize=10)
    ax_pred.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 예측 및 시각화 (검증 샘플 하나)
y_pred = model.predict(X_val_dense[:1])[0, ..., 0]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(y_val_dense[0, ..., 0], cmap='Reds')
plt.title('Actual (5x5)')

plt.subplot(1, 2, 2)
plt.imshow(y_pred, cmap='Reds')
plt.title('Predicted (5x5)')

plt.tight_layout()
plt.show()

# 손실 그래프 시각화
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# 예측 정확도 평가
# 검증 세트 전체에 대한 예측
y_val_pred = model.predict(X_val_dense)

# MSE 계산
from sklearn.metrics import mean_squared_error
val_mse = mean_squared_error(y_val_dense.reshape(-1), y_val_pred.reshape(-1))
print(f'검증 세트 MSE: {val_mse:.6f}')

# 예측값-실제값 산점도
plt.figure(figsize=(8, 8))
plt.scatter(y_val_dense.reshape(-1), y_val_pred.reshape(-1), alpha=0.5)
plt.plot([y_val_dense.min(), y_val_dense.max()], 
         [y_val_dense.min(), y_val_dense.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Prediction Accuracy')
plt.grid(True)
plt.show()