from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

#seed 설정
seed = 202112333
torch.manual_seed(seed)
np.random.seed(seed)

def train_model(model, train_loader, loss_function, optimizer, device, epochs=10):
  loss_storage = []

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total = 0  

    for batch_x, batch_y in train_loader:
      
      # 데이터를 GPU로 보내기
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)

      # Gradient 초기화
      optimizer.zero_grad()
      outputs = model(batch_x).view(-1)
      batch_y = batch_y.view(-1)

      # loss 값 계산
      loss = loss_function(outputs, batch_y)
      # 오류 역전파
      loss.backward()
      # optimizer 계산
      optimizer.step()

      running_loss += loss.item() * batch_x.size(0)

      total += batch_x.size(0)

    epoch_loss = running_loss / total
    loss_storage.append(epoch_loss)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

  # 최소 loss 값 
  print("Loss Min: ", np.min(loss_storage))
  
########################################################
#SiLU(x) = x * 시그모이드함수

class BinaryClassifier(nn.Module):
  def __init__(self):
    super(BinaryClassifier, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(2, 256),
        nn.SiLU(),
        nn.Linear(256, 128),
        nn.SiLU(),
        nn.Linear(128, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
        )

    # He 초기화
    for m in self.net:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # SiLU 포함
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

  def forward(self, x):
      return self.net(x)
########################################################
  
device = torch.device("cpu")
print(device)

# 데이터 읽어오기 & 저장
# dataA
project_dir = Path(__file__).resolve().parent
#df = pd.read_csv(project_dir / "dataA.csv")
#dataB
#df = pd.read_csv(project_dir / "dataB.csv")  
#dataC
df = pd.read_csv(project_dir / "dataC.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#StandardScaler로 정규화
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 제작한 model --> GPU로 보내기
model = BinaryClassifier().to(device)

# LOSS 함수
loss_function = nn.BCELoss()
# optimization 함수
#AdamW 이용
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 데이터들 Tensor화
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 데이터들 dataset 화
train_dataset = TensorDataset(x_tensor, y_tensor)
# 데이터 뽑아오는 함수 (배치 사용)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# TRAIN!!
train_model(model=model, 
            train_loader=train_loader, 
            loss_function=loss_function, 
            optimizer=optimizer, 
            device=device,
            epochs=100)
