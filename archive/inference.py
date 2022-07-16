import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import PIL
import os
from torch.autograd import Function
import cv2
import numpy as np
import torchsnooper
from utils import *
from gradcam import *
from torchvision.utils import make_grid, save_image

# Testing part
model = models.resnet34().to("cuda")
# model.fc = nn.Linear(model.fc.in_features, 2)
arch = models.resnet34().cuda()
arch.eval();

save_path = "./models/model.pt"
# test_path = "./testing/"
test_path = "/home/eragon/Desktop/Datasets/ArtClass/Popular/wlop/"
output_path = "./outputs/wlop/"
loc = "cuda:0"
# checkpoint = torch.load(save_path, map_location=loc)
# model.load_state_dict(checkpoint['net'])

model.eval()
print(f"Done loading pretrained")

print(model)

transform = transforms.ToTensor()

# @torchsnooper.snoop()
def grad_pred(test_path, i, pr):
    pil_img = PIL.Image.open(test_path+i)
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(
        2, 0, 1).unsqueeze(0).float().div(255).to(loc)
    torch_img = F.upsample(torch_img, size=(224, 224),
                           mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)

    cam_dict = {}
   # arch.fc = nn.Linear(arch.fc.in_features, 2)

    resnet_model_dict = dict(type='resnet', arch=arch,layer_name='layer4', input_size=(256, 256))
    # resnet_model_dict = dict(type='resnet', arch=model,layer_name='layer4', input_size=(224, 224))
    resnet_gradcam = GradCAM(resnet_model_dict, True)
    resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

    images = []
    for gradcam, gradcam_pp in cam_dict.values():
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.append(torch.stack(
            [torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

    images = make_grid(torch.cat(images, 0), nrow=5)
    save_image(images, f'./{output_path}/{pr}_{i}')


# @torchsnooper.snoop()
def predictor(inp_path):
    pred_dict = {0: 'Popular', 1: 'Unpopular'}
    img = Image.open(inp_path)
    img = transform(img).unsqueeze(0)
    output = model(img.to("cuda:0"))
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(output)
    pred = output.max(1, keepdim=True)
    output= None
    # retp = pred_dict[pred[1][0][0].detach().item()]
    print(f"Prediction : {retp}")
    return retp


# Testing on custom images
for i in tqdm(os.listdir(test_path)):
    if "git" not in i:
        i2 = test_path + i
        try:
            print(i)
            # pr = predictor(i2)
            pr = "Pop"
            grad_pred(test_path, i, pr)

        except Exception as e:
            print(e)
