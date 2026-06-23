import numpy as np
import torch
steps=20

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# tau 我设的是跟epsilon一样大
# inputs ndarray shape 是[3,224,224]
# label_inputs 是input 图片的ground truth label
# tau 这里设置的是跟epsilon一样大 效果还可以



tau = 0.05

def compute_ig(inputs,label_inputs,model):
    baseline = np.zeros(inputs.shape)
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in
                     range(0, steps + 1)]

    scaled_inputs = np.asarray(scaled_inputs)
    scaled_inputs = scaled_inputs + np.random.uniform(-tau,tau,scaled_inputs.shape)
    scaled_inputs = torch.from_numpy(scaled_inputs)
    scaled_inputs = scaled_inputs.to(device, dtype=torch.float)
    scaled_inputs.requires_grad_(True)
    att_out = model(scaled_inputs)
    score = att_out[:, label_inputs]
    loss = -torch.mean(score)
    model.zero_grad()
    loss.backward()
    grads = scaled_inputs.grad.data
    avg_grads = torch.mean(grads, dim=0)
    delta_X = scaled_inputs[-1] - scaled_inputs[0]
    integrated_grad = delta_X * avg_grads
    IG = integrated_grad.cpu().detach().numpy()
    del integrated_grad,delta_X,avg_grads,grads,loss,score,att_out
    return IG
