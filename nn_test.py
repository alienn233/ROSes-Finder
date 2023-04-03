import torch
from yyy_nn import Net # 请确保从同一文件导入Net类
from sklearn.metrics import accuracy_score

model = Net()
#model.load_state_dict(torch.load('/ifs1/User/yanyueyang/yyy/ROS/ref/step09/nn/xgboos_2class.pkl'))
#nn__2class.pkl
model.load_state_dict(torch.load('nn__2class.pkl'))
data={}
#data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2]
with open('DPC.out', 'r') as f:

    lines = f.readlines()
    lines = lines[1:]
    id=lines[0]
    for line in lines:
        items = line.strip().split("\t")
        name = items[0]
        features = [float(item) for item in items[1:]]
        data[name] = features
#tensor_data = torch.tensor([data], dtype=torch.float32)
tensor_data = torch.tensor(list(data.values()), dtype=torch.float32)
output = model(tensor_data)
predictions = torch.argmax(output, dim=1).tolist()
for i, name in enumerate(data.keys()):
#    print("样本 %s 的预测结果为: %d" % (name, predictions[i]))
    print(predictions[i])
