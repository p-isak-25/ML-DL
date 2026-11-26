#파이토치로 소프트맥스의 비용함수 구현(로우 레벨)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

#난수 고정시키기 위한 
torch.manual_seed(1)

#0~1 사이의 실수 값으로 3x5 tensor만들기
#역전파시 기울기를 계산해야함을 명시
z=torch.rand(3, 5, requires_grad = True)
#dim =1, 즉, **각 샘플(row(행))**마다 확률 분포로 변환됨
hypothesis = F.softmax(z, dim = 1)
print(hypothesis)

#0~4사이의 값들 중 무작위로 3개 생성
#(3,) : 1차원 텐서 + 요소 3개
#.long() 생성된 tensor타입을 long으로 변환환
y = torch.randint(5,(3,)).long()
print(y)


#소프트맥스 회귀의 비용함수 구현하기
#모든 원소가 0의 값을 가진 3x5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis)
print(y_one_hot)
#첫번쨰 인자 1로 행 단위로 수행하고,
#세번째 인자의 1을 y.unsqueeze(1)이 알려준위치에 넣어라
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()
print(cost)

#파이토치로 소프트맥스의 비용함수 구현하기(하이레벨)
#로우 레벨
#dim=0 → 열을 따라(세로 방향) 합
#dim=1 → 행을 따라(가로 방향) 합
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()

#위 아래 똑같은식, 

#하이 레벨
#F.log_softmax(z, dim = 1) = torch.log() + F.softmax
cost = (y_one_hot * F.log_softmax(z, dim =1)).sum(dim = 1).mean()

#더하이레벨
#nll: negative log likelihood
F.nll_loss(F.log_softmax(z, dim = 1), y)

#위에 F.nll_loss + F.log_softmax가
#F.cross_entropy이다. 
F.cross_entropy(z, y)