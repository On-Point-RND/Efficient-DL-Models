
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms



class InfernceIMG:
    
    """
    Class pefmorma matting with a model 
    combine returns two PIL images
    """
    def __init__(self,device, ref_size=512):
        
        self.ref_size = ref_size
        self.device = device
        self.im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
        
    def process_image(self, image):
        # unify image channels to 3
        im = np.asarray(image)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = self.im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < self.ref_size or min(im_h, im_w) > self.ref_size:
            if im_w >= im_h:
                im_rh = self.ref_size
                im_rw = int(im_w / im_h * self.ref_size)
            elif im_w < im_h:
                im_rw = self.ref_size
                im_rh = int(im_h / im_w * self.ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        return im, im_h, im_w
    
    def combine(self, matte, original):
    
        # calculate display resolution
        w, h = original.width, original.height
        rw, rh = 800, int(h * 800 / (3 * w))
        
        # obtain predicted foreground
        image = np.asarray(original)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
        foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
        
        # combine image, foreground, and alpha into one line
        combined = np.concatenate((image, foreground, matte * 255), axis=1)
        combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
        return combined, Image.fromarray((foreground).astype(np.uint8))
      
    def transform(self, model, image):
        im, im_h, im_w = self.process_image(image)
        im = im.to(self.device)
        _, _, matte = model(im.to(self.device), True)
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        
        return Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        