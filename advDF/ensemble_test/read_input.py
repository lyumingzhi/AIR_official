import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
# img1=cv2.imread('advDF/ensemble_test/input_image1.png')
# img2=cv2.imread('advDF/ensemble_test/input_image2.png')

# img1=np.load('advDF/ensemble_test/out1.npy')
# img2=np.load('advDF/ensemble_test/out2.npy')
# print(img1-img2)
# print('{:.20f}'.format(np.sum(img1-img2)))

Img1=cv2.imread('advDF/ensemble_test/ensemble_wb_test_transfer_attack_result_admin19_faceshifter.jpg')
Img2=cv2.imread('advDF/ensemble_test/result/ensemble_df_test_ensemble_fr_re_loss_18_34_50_100_hc_e=0.05_k=15_lr=0.02_lambda=2000_dataset=celebAHD_1000_mask=bg_grad_dynamic_preprocess_wo_bias_conv=13_transfer_attack_self_ver2/ensemble_wb_test_transfer_attack_result_admin19_faceshifter.jpg')
img1=transforms.ToTensor()(Img1[1024*4:1024*5,0:1024,:])

cv2.imwrite('advDF/ensemble_test/advsource.jpg',Img1[1024*4:1024*5,0:1024,:])
cv2.imwrite('advDF/ensemble_test/source.jpg',Img1[1024*0:1024*1,0:1024,:])

img2=transforms.ToTensor()(Img1[1024*0:1024*1,0:1024,:])

print((torch.abs(img2 -img1 )>0.02).detach().cpu().numpy().transpose(1,2,0)*255)
cv2.imwrite('advDF/ensemble_test/noise.jpg',(torch.abs(img2 -img1 )>0.02).detach().cpu().numpy().transpose(1,2,0)*255)
print(Img1.shape)
print(img2.size())
print(torch.max(img2-img1))
print(torch.sum(torch.abs(img2 -img1 )>0.02))