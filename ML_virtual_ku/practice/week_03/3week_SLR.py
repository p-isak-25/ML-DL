import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1) 


#data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#모델 초기화
#w,b를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시
W = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

#optimizer 설정
optimizer = optim.SGD([W,b], lr=0.01)

#기타변수: epoch설정
nb_epochs = 2000 #dataset을 몇번 돌릴건지 정함

for epoch in range(nb_epochs + 1):
    #H(x)
    hypothesis = x_train * W + b

    #loss
    loss = torch.mean((y_train - hypothesis)**2)

    #loss로 H(x) 개선
    optimizer.zero_grad() 
    #torch는 미분을 통해 얻은 기울기를 계속 누적하기 때문에 계속 반복 1번 돌때마다 0으로 초기화해야하낟.
    loss.backward()
    optimizer.step()

    if(epoch % 100) == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} loss: {:.6f}'.format(
        epoch, nb_epochs, W.item(), b.item(), loss.item()
    ))
        
print("---------------------------------------")        


#torch.nn 과 torch.nn.functional의 내장함수 이용
model = nn.Linear(1,1)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
for epoch in range(nb_epochs + 1):
    #H(x)
    prediction = model.forward(x_train)

    #loss
    loss = F.mse_loss(prediction, y_train) 
    
    #loss로 H(x) 개선
    optimizer.zero_grad() 
    #torch는 미분을 통해 얻은 기울기를 계속 누적하기 때문에 계속 반복 1번 돌때마다 0으로 초기화해야하낟.
    loss.backward()
    optimizer.step()

    if(epoch % 100) == 0:
        print('Epoch {:4d}/{} loss: {:.6f}'.format(
        epoch, nb_epochs, loss.item()
    ))


#class로 구현하기
class LinearModelRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self, x):
        return self.linear(x)
    
model = LinearModelRegression()
