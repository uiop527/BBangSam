#사람의 손글씨 MNIST를 이용해 Multi Layer Perceoptron 설계하기

#사용할 module import
import numpy as np
import matplot.pyplot as plt
import torch
import torch,nn as nn
import torch.nn.functional as F
from trchvision import transforms, datasets

#CUDA버전과 사용하는 DEVICE 출력
#PYTORCH역시 CPU버전을 받음
#노트북에 그래픽카드 미부착
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
print('Using Pytorch version : ', torch.__version__, 'Device: ', DEVICE)


BATCH_SIZE = 32     #BATCH_SIZE: MLP모델을 학습할 떄 필요한 데이터 개수의 단위, 딥러닝 모델에서 파라미터를 업데이트할 때 계산되는 데이터의 계수
EPOCHS =10          #EPOCHS: Mini-Batch 1개 단위로 Back Propagation을 이용해 가중값 업데이트 <= 여기서 Mini-Batch를 전부 사용하는 횟수

#사용할 MNIST데이터 다운로드 (MNIST: 사람 손글씨 데이터 베이스)
#Train set 과 Test set 분리
train_dataset = datasets.MNIST(root = "../data/MNIST", train = True, download = True, transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "../data/MNIST", train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

#데이터 확인하기