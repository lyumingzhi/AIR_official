import torch.nn
from torch.nn import parallel
from torch.nn.modules.module import Module
class Noise_Func(Module):
    def __init__(self,input_size,a,b):
        super(Noise_Func,self).__init__()
        self.a=None
        self.b=None

        if a!=None and b!=None:
            if len(input_size)==3:
                self.a=torch.nn.Parameter((torch.zeros((3,1,1))+a),requires_grad=True)
                self.b=torch.nn.Parameter((torch.zeros((3,1,1))+b),requires_grad=True)
            elif len(input_size)==4:
                self.a=torch.nn.Parameter((torch.zeros((1,3,1,1))+a),requires_grad=True)
                self.b=torch.nn.Parameter((torch.zeros((1,3,1,1))+b),requires_grad=True)
        self.delta_x=torch.nn.Parameter(torch.zeros(input_size),requires_grad=True)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,x,mask=None):
        # x=x.cuda()
        # mask=mask.cuda()
        # return self.a*x+self.b
        # print('mask',mask)
        # return self.delta_x*mask+x*((self.sigmoid(self.a)-0.5)*0.01+1)+(self.sigmoid(self.b)-0.5)*0.01
        if self.a!=None and self.b!=None:
            return self.delta_x*mask+x*(1+self.a.expand(x.size())*0.1)+self.b.expand(x.size())*0.1
        else:
            return self.delta_x*mask+x
    def update_func(self):
        if self.a !=None and self.b!=None:
            self.a.data=torch.clamp(self.a,min=-1,max=1).detach()
            self.b.data=torch.clamp(self.b,min=-1,max=1).detach()
        
if __name__=='__main__':
    noise=Noise_Func((1,3,112,112),1,0)
    noise.to('cuda')
    print(noise.parameters())