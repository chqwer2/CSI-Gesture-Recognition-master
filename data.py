import numpy as np
import torch

m = np.load("F:/Python深度学习/python_Project/IoT/DataSet/SignFi/amp_datas.npy")
n = np.load("F:/Python深度学习/python_Project/IoT/DataSet/SignFi/labels.npy")
print(np.shape(m))


amp_index_change = np.empty([ 3, 200, 30, 7500 ] ,dtype=float) #???大小调一下
from torch.utils.data import DataLoader

# [3,200,30]
from sklearn.model_selection import train_test_split

# for i in range(0, 7500):
for j in range(0, 3):
    amp_index_change[j, :, :, :] = m[:, :, j, :]

amp_index_Tensor = torch.from_numpy(amp_index_change)  # change the type from numpy to tensor
amp_index_Tensor = amp_index_Tensor.type(torch.FloatTensor)  # change to Float

print(amp_index_Tensor.size())
n = n.reshape(7500)
label_index_Tensor = n.astype(np.int)
label_index_Tensor = torch.tensor(label_index_Tensor)
label_index_Tensor = label_index_Tensor.type(torch.LongTensor)

# X_train, X_test, y_train, y_test = train_test_split(amp_index_Tensor, label_index_Tensor, test_size=0.1, random_state = 0)



train_dataset = torch.utils.data.TensorDataset(amp_index_Tensor, label_index_Tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10)  #


for data in train_loader:
    img, label = data
    img = Variable(img)
    img.shape()