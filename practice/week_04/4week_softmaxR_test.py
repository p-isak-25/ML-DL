import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#4개의 특성, 8개의 샘플
x_train = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5],
           [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]

#3개의 클래스스
y_train = [2,2,2,1,1,1,0,0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

#8x3
#3은 클래스의 개수
y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

W = torch.zeros((4, 3), requires_grad = True)

#브로드캐스팅
#행과 열중에 하나라도 1인곳이 있으면 그걸 더하는 행렬의
#행 또는 열에 맞춰 복사해서 더한다. 
b = torch.zeros((1, 3), requires_grad = True)

optimizer = optim.SGD([W,b], lr = 1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hypothesis = F.softmax(x_train.matmul(W) + b, dim = 1)

    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

#하이레벨로 구현
W = torch.zeros((4,3), requires_grad = True)
b = torch.zeros((1,3), requires_grad = True)

optimizer = optim.SGD([W, b], lr = 1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    #F.cross_entropy 안에 F.nll_loss(-torch.log) + F.softmax 다 들어있음
    cost = F.cross_entropy(x_train.matmul(W) + b, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


#nn.Module로 구현하기 
#4개의 특성을 가지고, 3개의 클래스로 분류
model = nn.Linear(4,3)

optimizer = optim.SGD(model.parameters(), lr = 0.1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    prediction = model(x_train)

    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


#class로 구현하기
class softmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
    
    def forward(self, x):
        return self.linear(x)
    

model = softmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr = 0.1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    prediction = model(x_train)

    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

