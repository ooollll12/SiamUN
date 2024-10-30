import torch
import torch.nn.functional as F
from net import SiamUN

def infer(model, input_data, device):
    # 将输入数据移动到设备
    input_data = input_data.to(device)

    # 进行推理
    with torch.no_grad():  # 禁用梯度计算
        output = model(input_data)  # 获取模型输出

    return output

device = torch.device('cuda' if (torch.cuda.is_available() ) else 'cpu')
anchors = {
    "stride": 8,
    "ratios": [0.33, 0.5, 1, 2, 3],
    "scales": [8],
    "round_dight": 0
}
model = SiamUN(anchors).to(device)
model.eval()

input_data = torch.randn(1, 3, 224, 224).to(device)  # 示例输入，替换为实际数据

# 获取模型输出
output = infer(model, input_data, device)

# 打印输出结果
print("Model output:", output)