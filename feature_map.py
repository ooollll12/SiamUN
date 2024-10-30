import torch
import torch.nn as nn
import random
import torchvision
from PIL import Image
from torchvision import transforms, models
from matplotlib import pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import numpy as np
#from model import resnet

model = torchvision.models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
#model = resnet()
#weights_path = r"C:\Users\Administrator\PycharmProjects\pythonProject/resnet50-19c8e357.pth"
#model.load_state_dict(torch.load(weights_path))
nodes, _ = get_graph_node_names(model)
#print(nodes)
#features = ['x', 'conv1', "layer4.0.conv1", 'layer2']
features = ['conv1']
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
original_img = Image.open("image.jpg")

img = transform(original_img).unsqueeze(0)

feature_extractor = create_feature_extractor(model, return_nodes=features)

out = feature_extractor(img)


def draw_feature_map(out, features):
    for j in range(len(features)):

        feature_map_data_list = [out[features[j]][0, i].detach().numpy() for i in range(out[features[j]].shape[1])]
        feature_map_data_ = []
        for i in range(9):
            n = random.randint(0, out[features[j]].shape[1] - 1)
            feature_map_data_.append(feature_map_data_list[n])

        for i, feature_map_data in enumerate(feature_map_data_):
            plt.subplot(3, 3, i + 1)
            #plt.imshow(feature_map_data, cmap="viridis")
            plt.imshow(feature_map_data, cmap="grey")
            plt.title(f"{i + 1}")
            plt.axis('off')
        plt.show()



def save_feature_map(out, features):
    for j in range(len(features)):

        feature_map_data_list = [out[features[j]][0, i].detach().numpy() for i in range(out[features[j]].shape[1])]
        feature_map_data_ = []
        for i in range(4):
            n = random.randint(0, out[features[j]].shape[1] - 1)
            feature_map_data_.append(feature_map_data_list[n])

        for i, feature_map_data in enumerate(feature_map_data_):
            # 归一化数据到0-255范围
            normalized_data = ((feature_map_data - feature_map_data.min()) / (feature_map_data.max() - feature_map_data.min()) * 255).astype(np.uint8)
            
            # 创建PIL图像
            img = Image.fromarray(normalized_data)
            
            # 保存特征图
            img.save(f'{features[j]}_{i+1}.png')

if __name__ == "__main__":
    #draw_feature_map(out, features)
    save_feature_map(out, features)

