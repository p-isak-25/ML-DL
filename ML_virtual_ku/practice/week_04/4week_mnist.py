# mnist 문제: 손글씨로 적힌 숫자를 이미지로 입력받으면, 어떤 숫자인지 맞추는 문제
# mnist 개념념
# 숫자 0~9의 이미지로 이루어진 손글씨 데이터셋
# 6만개의 훈련데이터셋과 1만개의 테스트 데이터셋으로 구성
# 레이블은 0~9까지 총 10개

#mnist image: 784개의 픽셀로 구성
# 28x28
# 전처리과정에서 784개의 원소를 가진 벡터로 만든다. 


# 토치비전 패키지: 파이토치에서 이미지 관련 작업을 쉽게 해주는 라이브러리
# datasets(유명 데이터셋들들), transforms(이미지 전처리 도구), models(이미 구현되어있는 유명한 모델들)

from pathlib import Path

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

device = torch.device("cpu")
print(device)

random.seed(777)
torch.manual_seed(777)

training_epochs = 15
batch_size  = 100 

data_dir = Path(__file__).resolve().parent / "MNIST_data"
mnist_train = dsets.MNIST(root = str(data_dir), train = True, 
                        transform = transforms.ToTensor(), download = True)

mnist_test = dsets.MNIST(root = str(data_dir), train = False,
                        transform = transforms.ToTensor(), download = True)


data_loader = DataLoader(dataset = mnist_train,
                         batch_size = batch_size,
                         shuffle = True,
                         drop_last = True)

linear = nn.Linear(784, 10, bias = True).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        #배치크기가 100이므로 아래의 연산에서 x는 (100, 784)의 텐서가 된다. 
        X = X.view(-1, 28 * 28).to(device)

        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' %(epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
   
print("learning finished")

with torch.no_grad():
    X_test = mnist_test.data.view(-1, 28*28).float().div(255).to(device)
    Y_test = mnist_test.targets.to(device)
    prediction = linear(X_test)
    #dim = 1 행마다 확률이 가장 높은 클래스의 인덱스를 반환
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy:", accuracy.item())

    #MNIst 데이터에서 무작위로 하나를 뽀ㅃ아서 예측한다.
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.data[r:r + 1].view(-1, 28 * 28).float().div(255).to(device) 
    Y_single_data = mnist_test.targets[r:r + 1].to(device) 
    print('Label: ', Y_single_data.item()) 
    single_prediction = linear(X_single_data) 
    print('Prediction: ', torch.argmax(single_prediction, 1).item()) 

    plt.imshow(mnist_test.data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()