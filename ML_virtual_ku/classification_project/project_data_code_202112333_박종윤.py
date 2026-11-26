from pathlib import Path

from random import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 복잡한 부등식 정의 (주기 다르게 설정 + 대각선 성분 포함)
def inequality(x, y):
    return (
       np.sin(x / 1300) +
       np.cos(y / 2700) +
       0.3 * np.sin((x + y) / 800)) > 0.7

# 2. 데이터 생성 함수
def generate_data(num_samples=5000, noise_ratio=0.1, x_range=(-10000, 10000), y_range=(-10000, 10000)):
    x_all = []
    y_all = []
    labels_all = []

    count_0, count_1 = 0, 0
    while count_0 < num_samples or count_1 < num_samples:
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        label = 1 if inequality(x, y) else 0

        if label == 1 and count_1 < num_samples:
            x_all.append(x)
            y_all.append(y)
            labels_all.append(label)
            count_1 += 1
        elif label == 0 and count_0 < num_samples:
            x_all.append(x)
            y_all.append(y)
            labels_all.append(label)
            count_0 += 1

   # 3. 라벨별로 5%씩 반전
    total_samples = num_samples * 2
    label_0_indices = [i for i, l in enumerate(labels_all) if l == 0]
    label_1_indices = [i for i, l in enumerate(labels_all) if l == 1]

    noise_0 = np.random.choice(label_0_indices, int(0.05 * total_samples), replace=False)
    noise_1 = np.random.choice(label_1_indices, int(0.05 * total_samples), replace=False)

    for i in np.concatenate([noise_0, noise_1]):
        labels_all[i] = 1 - labels_all[i]

    return np.array(x_all), np.array(y_all), np.array(labels_all)


# 4. 데이터 생성 및 저장
project_dir = Path(__file__).resolve().parent
x_all, y_all, labels_all = generate_data()
df = pd.DataFrame({'x': x_all, 'y': y_all, 'label': labels_all})
df.to_csv(project_dir / "project_data_202112333_박종윤.csv", index=False, encoding="utf-8-sig")

# 5. 시각화 함수
def plot_data(filename):
    df = pd.read_csv(filename)

    plt.figure(figsize=(10, 8))
    plt.scatter(df[df['label'] == 0]['x'], df[df['label'] == 0]['y'], color='blue', label='Label 0(sin(x/1300) + cos(y/2700) + 0.3*sin((x+y)/800) <= 0.7)', alpha=0.5)
    plt.scatter(df[df['label'] == 1]['x'], df[df['label'] == 1]['y'], color='red', label='Label 1(sin(x/1300) + cos(y/2700) + 0.3*sin((x+y)/800) > 0.7)', alpha=0.5)

    plt.title("sin(x/1300) + cos(y/2700) + sin((x+y)/800)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.savefig(project_dir / "project_data_image_202112333_박종윤.png")
    plt.show()

# 6. 시각화 실행
plot_data(project_dir / "project_data_202112333_박종윤.csv")
