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