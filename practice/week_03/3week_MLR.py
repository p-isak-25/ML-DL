import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [80], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

W1 = torch.zeros(1, requires_grad = True)
W2 = torch.zeros(1, requires_grad = True)
W3 = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W1, W2, W3, b], lr = 1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypothesis = W1 * x1_train + W2 * x2_train + W3 * x3_train + b

    loss = torch.mean((y_train - hypothesis) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
      epoch, nb_epochs, W1.item(), W2.item(), W3.item(), b.item(), loss.item()
    ))

print("--------------------" * 10)

#행렬연산으로 구현(내적이용)
x_train = torch.FloatTensor([[73, 80, 75], 
                             [93, 88, 93], 
                             [89, 91, 80], 
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

W = torch.zeros((3, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W, b], lr = 1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    #x_train과 W행렬의 내적 계산산 
    #matrix multiplication: x_train.matmul(W)
    hypothesis = x_train.matmul(W) + b

    loss = torch.mean((y_train - hypothesis) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
      epoch, nb_epochs, hypothesis.squeeze().detach(), loss.item()
    ))

print("----------------" * 5)
#torch.nn과 torch.nn.Functional의 내장함수 이용
model = nn.Linear(3,1)

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    #x_train과 W행렬의 내적 계산산 
    #matrix multiplication: x_train.matmul(W)
    prediction = model.forward(x_train)

    loss = F.mse_loss(prediction, y_train)

    optimizer.zero_grad() #torch에선 미분한 기울기들이 누적되기때문에 계속 0으로 초기화해야함
    loss.backward() #loss_function을 미분해서 기울기를 구하기
    optimizer.step() #W와 b 업데이트

    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
      epoch, nb_epochs, loss.item()
    ))
