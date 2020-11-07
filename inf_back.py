import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

model = models.resnet34().to("cuda")
save_path = "./models/model.pt"

loc = "cuda:0"
checkpoint = torch.load(save_path, map_location = loc)
model.load_state_dict(checkpoint['net'])

model.eval();
print(f"Done loading pretrained")

transform = transforms.ToTensor()

def predictor(inp_path):
    pred_dict = {0:'Popular', 1:'Unpopular'}
    img = Image.open(inp_path)
    img = transform(img).unsqueeze(0)
    output = model(img.to("cuda"))
    sm = torch.nn.Softmax(dim = 1)
    probabilities = sm(output)
    pred = output.max(1, keepdim = True)
    print(f"Prediction : {pred_dict[pred[1][0][0].detach().item()]}")

# Testing on custom images
test_path = "./testing/"
for i in tqdm(os.listdir(test_path)):
    i = test_path+ i
    try:
        print(i, predictor(i))
    except :
        pass
