"""Autograd"""

import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

BATCH_SIZE=64       #
INPUT_SIZE=1000     #
HIDDEN_SIZE=100     #
OUTPUT_SIZE=10      #

x= torch.randn(BATCH_SIZE,INPUT_SIZE, device = DEVICE, dtype = torch.float,requires_grad = False)
y= torch.randn(BATCH_SIZE,OUTPUT_SIZE, device = DEVICE, dtype = torch.float,requires_grad = False)
w1= torch.randn(INPUT_SIZE,HIDDEN_SIZE, device = DEVICE, dtype = torch.float,requires_grad = True)
w2= torch.randn(HIDDEN_SIZE,OUTPUT_SIZE, device = DEVICE, dtype = torch.float,requires_grad = True)

learning_rate=1e-6
for t in range(1,501):
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    
    loss=(y_pred-y).pow(2).sum()
    if t % 100 ==0:
        print("Iteration: ", t, "\t", "Loss: ", loss.item())
    loss.backward()
    
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        w1.grad.zero_()
        w2.grad.zero_()