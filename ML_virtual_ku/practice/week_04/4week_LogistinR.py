import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# 파이토치로 LR구현하기
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# W,b를 0으로 초기화해주고, 학습에 의해 변경되는 변수임을 명시
W = torch.zeros((2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W,b], lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    cost = (-((y_train) * torch.log(hypothesis) + (1-y_train) * torch.log(1-hypothesis))).mean()

    #cost로 H(X) 개선시키는 과정
    optimizer.zero_grad() #torch는 기울기를 미분할때마다 누적하기때문에 반복할때마다 0으로 초기화해준다. 
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch: {:4d} / {} Cost: {:6f}".format(
            epoch, nb_epochs, cost.item()
        ))


#nn.module로 LR 구현하기
#nn.Sequential(): nn.module 층을 차례로 쌓아 올릴수 있도록 한다.  
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

for epoch in range(nb_epochs + 1):

    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    #cost로 H(X) 개선시키는 과정
    optimizer.zero_grad() #torch는 기울기를 미분할때마다 누적하기때문에 반복할때마다 0으로 초기화해준다. 
    cost.backward()
    optimizer.step()

    if epoch % 20 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print("Epoch: {:4d} / {} Cost: {:6f} Accuracy: {:2.2f}".format(
            epoch, nb_epochs, cost.item(), accuracy * 100
        ))


#class로 LR구현하기
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
    

model = BinaryClassifier()

for epoch in range(nb_epochs + 1):

    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    #cost로 H(X) 개선시키는 과정
    optimizer.zero_grad() #torch는 기울기를 미분할때마다 누적하기때문에 반복할때마다 0으로 초기화해준다. 
    cost.backward()
    optimizer.step()

    if epoch % 20 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print("Epoch: {:4d} / {} Cost: {:6f} Accuracy: {:2.2f}".format(
            epoch, nb_epochs, cost.item(), accuracy * 100
        ))

