import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class my_dataset(Dataset):
    def __init__(self, data_address, label_address):
        self.data = data_address
        self.label = label_address
        self.images = os.listdir(data_address)
    def __getitem__(self, index):
        image_item = self.images[index]
        img = plt.imread(os.path.join(self.data, image_item))
        return img

    def __len__(self):
        return len(self.data)


# 创建一个数据集


train_data = my_dataset(r'C:\Users\Administrator\Desktop\images',
                        r'C:\Users\Administrator\Desktop\lables')
# 创建一个dataloader,设置批大小为64，每一个epoch重新洗牌不舍弃不能被整除的批次
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2, drop_last=False)

print(train_loader.dataset.data)
print("get_item",train_data.__getitem__(0))