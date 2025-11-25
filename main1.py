import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# 기본 설정
data_dir = './miniData'
dates = [f'202202{day:02d}' for day in range(1, 29)]
hours = [f'{h:02d}' for h in range(24)]

# shape 확인용
sample_path = os.path.join(data_dir, f'{dates[0]}_m_2529_P_T_00.csv')
sample = np.loadtxt(sample_path, delimiter=',')
height, width = sample.shape  # 보통 72, 49

# 학습용 데이터 (1~27일)
X_train = np.zeros((27, 24, height, width))
for d_idx, date in enumerate(dates[:27]):
    for h_idx, hour in enumerate(hours):
        file_path = os.path.join(data_dir, f'{date}_m_2529_P_T_{hour}.csv')
        if os.path.exists(file_path):
            X_train[d_idx, h_idx] = np.loadtxt(file_path, delimiter=',')
        else:
            print(f'❌ 누락 파일: {file_path}')

# 테스트용 데이터 (28일)
y_test = np.zeros((24, height, width))
for h_idx, hour in enumerate(hours):
    file_path = os.path.join(data_dir, f'20220228_m_2529_P_T_{hour}.csv')
    if os.path.exists(file_path):
        y_test[h_idx] = np.loadtxt(file_path, delimiter=',')
    else:
        print(f'❌ 누락 파일: {file_path}')

print('X_train shape:', X_train.shape)  # (27, 24, 72, 49)
print('y_test shape:', y_test.shape)    # (24, 72, 49)

# 슬라이딩 윈도우 설정
window_size = 3  # 과거 3시간 → 다음 1시간 예측

X = []
y = []

# 슬라이딩 윈도우 데이터 생성
for day in range(X_train.shape[0]):
    for t in range(24 - window_size):
        input_window = X_train[day, t:t+window_size]  # (3, 72, 49)
        target = X_train[day, t+window_size]          # (72, 49)
        X.append(input_window)
        y.append(target)

X = np.array(X)  # shape: (samples, 3, 72, 49)
y = np.array(y)  # shape: (samples, 72, 49)

# CNN 입력 형태로 변환 (samples, height, width, channels)
X = X.transpose(0, 2, 3, 1)  # (samples, 72, 49, 3)
y = y[..., np.newaxis]      # (samples, 72, 49, 1)

# 학습/검증 분리
X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN 모델 정의
model = Sequential([
    Input(shape=(72, 49, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(72 * 49, activation='linear'),
    Reshape((72, 49, 1))
])

model.compile(optimizer=Adam(), loss='mse')

# 모델 학습
history = model.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn),
                    epochs=10, batch_size=16)

# 예측 및 시각화 (검증 데이터 중 하나)
y_pred = model.predict(X_val_cnn[:1])[0, ..., 0]  # shape: (72, 49)

# 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(y_val_cnn[0, ..., 0], cmap='Reds')
plt.title('Actual')

plt.subplot(1, 2, 2)
plt.imshow(y_pred, cmap='Reds')
plt.title('Predicted')

plt.tight_layout()
plt.show()