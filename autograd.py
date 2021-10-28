"""Autograd: Back Propagation을 이용해 파라미터를 업데이트 하는 방법 """

import torch
'''
Device가 어떤 프로세서를 사용하는지 알려줌
현재 colab에서 돌리는 것이 아니기 때문에 DEVICE=cpu라고 인식이 됨
pytorch 역시 cpu버전을 받음

추후 colab에서 cuda와 cuDNN을 import해서 사용할 예정
'''
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

###############################################

BATCH_SIZE=64       #딥러닝 모델에서 파라미터를 업데이할 때 계산되는 데이터 갯수 => Batch size수만큼 데이터를 이용해서 output을
                    #계산, 그 수만큼 오찻값을 계산
INPUT_SIZE=1000     #Input의 크기, 입력층의 노드 수, 1000으로 설정했으므로 입력데이터의 크기가 1000이라는 의미
#batch size와 input size를 통해서 1000크기의 벡터값을 64개 이용한다는 의미
HIDDEN_SIZE=100     #input을 파라미터를 이용해서 계산한 결과에 한번 더 계산하는 파라미터 수 = 입력층에서 은닉층으로 전달됐을 때
                    #은닉층의 노드 수
OUTPUT_SIZE=10      #딥러닝 모델에서 최종으로 출력되는 값의 벡터 크기 = 보통 최종으로 비교하고자 하는 레이블의 크기와 동일하게 설정


'''
데이터와 파라미터 설정
randn 은 평균이 0, 표준편차가 1인 정규분포에서 샘플링한 값 = 데이터를 만든다는 것을 의미, 데이터를 만들어낼 때 크기 설정 가능
gradient는 파라미터 값을 업데이트하기 위해 구하는 값
'''
x= torch.randn(BATCH_SIZE,INPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = False)
#크기가 1000짜리 벡터 64개로 설정하고 device는 이전에 설정한 device, 데이터 타입은 float, input으로 사용하기 때문에 gradient설정 X
#(64,1000)의 데이터 생성

y= torch.randn(BATCH_SIZE,OUTPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = False)
#output, batch size만큼의 결과값이 필요, 오차를 계산하기위해 10으로 설정 <= 이 부분은 아직 이해가 덜 됨

w1= torch.randn(INPUT_SIZE,HIDDEN_SIZE, device = DEVICE, dtype = torch.float, requires_grad = True)
#업데이트할 파라미터 값, x와의 행렬 곱을 하기 위해서는 행의 값이 1000이여야 하고 100은 데이터를 생성하고자 하는 크기 
#(1000,100)의 파라미터 값 생성

w2= torch.randn(HIDDEN_SIZE,OUTPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = True)
#w1과 x를 행렬 곱한 결과에 계산할 수 있는 데이터, x와 w1의 행렬 곱을 한 결과 (64,100)의 행렬이 나오고 이를 (100,10)행렬 곱으로
#output을 계산

learning_rate=1e-6 
#Gradient를 계산한 결과값에 1보다 작은 값을 곱해 업데이트하는데 이를 learning rate라고 한다.
#learning rate에 값에 따라 gradient값에 따른 학습 정도가 달라짐
#제일 중요한 Hyperparameter

for t in range(1,501):      #500번 반복을 위한 반복문
    
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    #딥러닝 모델의 결과값을 예측값이라고 한다. clamp(import torch): 비선형 함수 적용
    #y_pred = y 즉 output의 prediction value
    loss=(y_pred-y).pow(2).sum()
    #loss= 오차값, pow(2) = 제곱차를 의미 , 그것을 sum()으로 모두 더한다
    
    if t % 100 ==0:
        print("Iteration: ", t, "\t", "Loss: ", loss.item())
    #100번째마다 값을 보여주기 위한 함수
    
    loss.backward()
    #오차값에 대해서 backward()메소드를 사용하면 파라미터 값에 대해 gradient를 계산하고 back propagation을 진행한다는 뜻 
    
    with torch.no_grad():   #gradient값을 고정한다
        w1 -= learning_rate * w1.grad      #w1.grad = w1의 gradient값, 음수를 이용하는 이유는 오차값이 최소로 계산될 수 있는
        w2 -= learning_rate * w2.grad      #파라미터 값을 찾기 위해 gradient값에 대한 반대 방향으로 계산하는 것을 의미
        
        w1.grad.zero_()         #파라미터값을 업데이트 한 후 gradient를 초기화 하여 다음 반복문을 돌릴 수 있도록 gradient=0으로 설
        w2.grad.zero_()