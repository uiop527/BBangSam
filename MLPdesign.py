#사람의 손글씨 MNIST를 이용해 Multi Layer Perceoptron 설계하기

#사용할 module import
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

#CUDA버전과 사용하는 DEVICE 출력
#PYTORCH역시 CPU버전을 받음
#노트북에 그래픽카드 미부착
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
print('Using Pytorch version : ', torch.__version__, 'Device: ', DEVICE)


BATCH_SIZE = 32     #BATCH_SIZE: MLP모델을 학습할 떄 필요한 데이터 개수의 단위, 딥러닝 모델에서 파라미터를 업데이트할 때 계산되는 데이터의 계수
EPOCHS =10          #EPOCHS: Mini-Batch 1개 단위로 Back Propagation을 이용해 가중치 업데이트 <= 여기서 Mini-Batch를 전부 사용하는 횟수

#사용할 MNIST데이터 다운로드 (MNIST: 사람 손글씨 데이터 베이스)
#Train set 과 Test set 분리
train_dataset = datasets.MNIST(root = "../data/MNIST", train = True, download = True, transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "../data/MNIST", train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

#데이터 확인하기1
for(X_train, y_train) in train_loader:
    print('X_train: ', X_train.size(), 'type: ', X_train.type())
    print('y_train: ', y_train.size(), 'type: ', y_train.type())
    break
#X_train: torch.Szie([32, 1, 28, 28]) type: torch.FloatTensor
#y_train: torch.Szie([32]) type: torch.LongTensor

#데이터 확인하기2
pltsize = 1
plt.figure(figsize = (10 * pltsize, pltsize))

for i in range(10):
    plt.subplot(1, 10, 1 +i)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28,28), cmap = 'gray_r')
    plt.title('Class: ' + str(y_train[i].item()))
    
#MLP(Multi Layer Perception)모델 설계하기
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,10)
        self.dropout_prob= 0.5
        
    def forward(self,x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x
    
#Optimizer, Objective Function설정하기
model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()

print(model)
'''
Net(
    (fc1)Linewar(in_feature=784, out_feature = 512, bias = True)
    (fc2)Linewar(in_feature=512, out_feature = 256, bias = True)
    (fc3)Linewar(in_feature=256, out_feature = 10, bias = True)
    )    
'''

#MLP모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx,(image,label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(Epoch, batch_idx * len(image), 
                  len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            
#학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의
def evaluate(moel, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

#MLP학습을 실행하면서 Train, Test set의 Loss 및 Test set Accuracy확인하기
for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(Epoch, test_loss, test_accuracy))
