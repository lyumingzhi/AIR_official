import torch.nn
from torch.nn import parallel
from torch.nn.modules.module import Module
import cv2

class Contrast_func(Module):
    def __init__(self,input_imgs,bin_num,device):
        super(Contrast_func,self).__init__()
        self.device=device
        self.bin_num=bin_num

        pixel_mask_for_batch=[]
        pixel_middle_for_batch=[]

        self.unique_pixel_for_batch=[]
        upper_bound_pixels_for_batch=[]
        lower_bound_pixels_for_batch=[]
        for index in range(input_imgs.size()[0]):
            pixel_mask_for_img=[]
            pixel_middle_for_img=[]
            unique_pixel_for_img=[]
            upper_bound_pixels_for_img=[]
            lower_bound_pixels_for_img=[]
            for channel_index in range(input_imgs.size()[1]):
                unique_pixel_value=torch.unique(input_imgs[index,channel_index,:,:],sorted=True)
                bin_size=unique_pixel_value.size()[0]//self.bin_num
                assert unique_pixel_value.size()[0]>self.bin_num

                unique_pixel_for_img.append(unique_pixel_value)

                pixel_mask_for_channel=[]
                pixel_middle_for_channel=[]
                upper_bound_pixels_for_channel=[]
                lower_bound_pixels_for_channel=[]
                for i in range(self.bin_num):
                    pixel_mask_for_bin=0
                    if i!=self.bin_num-1:
                        for j in range(i*bin_size,(i+1)*bin_size):
                            tmp_mask=(input_imgs[index,channel_index,:,:]==unique_pixel_value[j])
                            pixel_mask_for_bin+=tmp_mask
                        middle_value=unique_pixel_value[int((i+i+1)*bin_size/2)]
                        upper_bound=unique_pixel_value[(i+1)*bin_size-1]
                        lower_bound=unique_pixel_value[i*bin_size]
                    else:
                        for j in range(i*bin_size,unique_pixel_value.size()[0]):
                            tmp_mask=(input_imgs[index,channel_index,:,:]==unique_pixel_value[j])
                            pixel_mask_for_bin+=tmp_mask
                        middle_value=unique_pixel_value[int((i*bin_size+unique_pixel_value.size()[0])/2)]
                        upper_bound=unique_pixel_value[-1]
                        lower_bound=unique_pixel_value[i*bin_size]
                    pixel_mask_for_channel.append(pixel_mask_for_bin.unsqueeze(0))
                    middle_value=torch.Tensor([middle_value])
                    pixel_middle_for_channel.append(middle_value)

                    upper_bound=torch.Tensor([upper_bound])
                    upper_bound_pixels_for_channel.append(upper_bound)

                    lower_bound=torch.Tensor([lower_bound])
                    lower_bound_pixels_for_channel.append(lower_bound)
                print(unique_pixel_value.size())
            
                
                assert torch.sum(torch.cat(pixel_mask_for_channel,dim=0))==input_imgs.size()[-1]*input_imgs.size()[-2]
                pixel_mask_for_channel=torch.cat(pixel_mask_for_channel,dim=0)
                # print('channel',pixel_mask_for_channel.size())
                pixel_mask_for_img.append(pixel_mask_for_channel.unsqueeze(0))

                pixel_middle_for_channel=torch.cat(pixel_middle_for_channel,dim=0)

                pixel_middle_for_img.append(pixel_middle_for_channel.unsqueeze(0))

                upper_bound_pixels_for_channel=torch.cat(upper_bound_pixels_for_channel,dim=0)
                upper_bound_pixels_for_img.append(upper_bound_pixels_for_channel.unsqueeze(0))

                lower_bound_pixels_for_channel=torch.cat(lower_bound_pixels_for_channel,dim=0)
                lower_bound_pixels_for_img.append(lower_bound_pixels_for_channel.unsqueeze(0))
            lower_bound_pixels_for_img=torch.cat(lower_bound_pixels_for_img,dim=0)
            lower_bound_pixels_for_batch.append(lower_bound_pixels_for_img.unsqueeze(0))

            upper_bound_pixels_for_img=torch.cat(upper_bound_pixels_for_img,dim=0)
            upper_bound_pixels_for_batch.append(upper_bound_pixels_for_img.unsqueeze(0))
            
            pixel_middle_for_img=torch.cat(pixel_middle_for_img,dim=0)
            pixel_middle_for_batch.append(pixel_middle_for_img.unsqueeze(0))
            # exit()
            pixel_mask_for_img=torch.cat(pixel_mask_for_img,dim=0)
            # print('img',pixel_mask_for_img.size())

            pixel_mask_for_batch.append(pixel_mask_for_img.unsqueeze(0))

            self.unique_pixel_for_batch.append(unique_pixel_for_img)


        pixel_middle_for_batch=torch.cat(pixel_middle_for_batch,dim=0)
        pixel_mask_for_batch=torch.cat(pixel_mask_for_batch,dim=0)
        upper_bound_pixels_for_batch=torch.cat(upper_bound_pixels_for_batch,dim=0)
        lower_bound_pixels_for_batch=torch.cat(lower_bound_pixels_for_batch,dim=0)
        # print(pixel_middle_for_batch.size())
        # print(input_imgs.size()[:3])
        assert torch.sum(pixel_mask_for_batch)==input_imgs.size()[-1]*input_imgs.size()[-2]*input_imgs.size()[-3]*input_imgs.size()[-4]
        assert pixel_middle_for_batch.size()==pixel_mask_for_batch.size()[:3]

        self.pixel_mask_for_batch=pixel_mask_for_batch
        self.pixel_middle_for_batch=pixel_middle_for_batch
        self.contrast_parameter_a=torch.nn.Parameter(torch.ones(self.pixel_mask_for_batch.size()[0:3]),requires_grad=True)
        self.contrast_parameter_b=torch.nn.Parameter(self.pixel_middle_for_batch.clone(),requires_grad=True)
        self.upper_bound_pixels_for_batch=upper_bound_pixels_for_batch.cuda()
        self.lower_bound_pixels_for_batch=lower_bound_pixels_for_batch.cuda()
        # self.contrast_parameter_a= torch.ones(self.pixel_mask_for_batch.size()[0:3])
        # self.contrast_parameter_b= self.pixel_middle_for_batch
        # print(self.pixel_mask_for_batch.size())
        print(self.upper_bound_pixels_for_batch)
        print(self.unique_pixel_for_batch[0][0].size())
        # exit()
    def forward(self,x,mask=None):
        print((self.pixel_mask_for_batch).size(),self.contrast_parameter_a.size(),self.contrast_parameter_b.size())
        contrast_adjust_mul=((self.pixel_mask_for_batch).cuda()*(self.contrast_parameter_a.unsqueeze(-1).unsqueeze(-1)*0+torch.ones(self.pixel_mask_for_batch.size()).cuda()*0.5)).sum(dim=2)*1.5*0
        contrast_adjust_add=((self.pixel_mask_for_batch).cuda()*self.contrast_parameter_b.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)


        adjusted_img=(x-contrast_adjust_add)*(contrast_adjust_mul)+contrast_adjust_add

        # cv2.imwrite('adjusted_img.jpg',adjusted_img.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255)

        return adjusted_img
        
    def update_func(self):
        
        self.contrast_parameter_b.data=torch.max(self.contrast_parameter_b,self.lower_bound_pixels_for_batch)
        self.contrast_parameter_b.data=torch.min(self.contrast_parameter_b,self.upper_bound_pixels_for_batch)

        self.contrast_parameter_a.data=torch.max(self.contrast_parameter_a,torch.ones(self.contrast_parameter_a.size()).to(self.device)*0)
        self.contrast_parameter_a.data=torch.max(self.contrast_parameter_a,torch.ones(self.contrast_parameter_a.size()).to(self.device)*1.5)


if __name__=='__main__':
    
    # print(noise.parameters())
    import cv2
    img=cv2.imread('src_test.jpg')
    import torchvision.transforms as transforms
    transformer=transforms.Compose([transforms.ToTensor()])
    img=transformer(img).unsqueeze(0)
    contrast=Contrast_func(img,3)
    contrast.to('cuda')

    contrast(img.cuda())