from __future__ import absolute_import, print_function
"""
SignFidata
"""
import torch
import torch.utils.data as data

import os
import sys
import numpy as np


def default_loader(root):   #root is the data storage path  eg. path = D:/CSI_Data/signfi_matlab2numpy/
    label_path = root + "/labels.npy"    #storage path : label of Sign
    amp_path = root + "/amp_datas.npy"   #storage path : the amplitude of CSI
    phase_path = root + "/phase_datas.npy" #storage path :  the phase value of CSI
    label_env_path = root + "/label_env.npy" #storage path : label of person or scenario
    label_all = np.load(label_path)     #label of Sign
    label_env_all = np.load(label_env_path) #lable of env(person or scenario)
    amp_all = np.load(amp_path)
    phase_all = np.load(phase_path)
    return label_all,label_env_all,amp_all,phase_all


class MyData(data.Dataset):
    def __init__(self, root, set_name, loader = default_loader):  #set_name = "test" or "train" ;  root is the path of data;

        self.root = root
        self.load = loader
        label_all,label_env_all,amp_all,phase_all = self.load(root)
        if set_name == "train":
            self.label = label_all[0:4500]
            self.label_env = label_env_all[0:4500]
            self.amp = amp_all[:,:,:,0:4500]
            self.phase = phase_all[:,:,:,0:4500]
        else :
            self.label = label_all[4500:7500]
            self.label_env = label_env_all[4500:7500]
            self.amp = amp_all[:,:,:,4500:7500]
            self.phase = phase_all[:,:,:,4500:7500]

    def __getitem__(self, index):
        label_index,label_env_index, amp_index, phase_index = self.label[index],self.label_env[index], self.amp[:,:,:,index], self.phase[:,:,:,index]
        # choosing amplitude or phase values of CSI for perception
        amp_index_change = np.empty([3,200,30],dtype= float) #the initial amp.shape is [200,30,3] change to [3,200,30]
        phase_index_change = np.empty([3,200,30],dtype= float)
        for i in range(0,3):
            amp_index_change[i,:,:] = amp_index[:,:,i]
            phase_index_change[i,:,:] = phase_index[:,:,i]
        amp_index_Tensor = torch.from_numpy(amp_index_change)    # change the type from numpy to tensor
        amp_index_Tensor = amp_index_Tensor.type(torch.FloatTensor) #change to Float
        phase_index_Tensor = torch.from_numpy(phase_index_change)
        phase_index_Tensor = phase_index_Tensor.type(torch.FloatTensor)  # change to Float

        label_index_Tensor = label_index.astype(np.int)
        label_index_Tensor = torch.tensor(label_index_Tensor)
        label_index_Tensor = label_index_Tensor.type(torch.FloatTensor)



        #return data
        return amp_index_Tensor,label_index_Tensor   #or output phase_index_Tensor   and   label_env

    def __len__(self):
        return len(self.label_env)

class SignFi:
    def __init__(self, root):
            self.train = MyData(root, set_name="train")
            self.test = MyData(root, set_name = "test")


def test_SignFi(path):
    print("hahahahhahahha")
    print(SignFi.__name__)
    data = SignFi(path)
    print("the length of training set:{}".format(len(data.train)))
    print("the length of test set:{}".format(len(data.test)))
    print("the data of train[0]:{}".format(data.train[0]))
    print("the data of train[3000]:{}".format(data.train[3000]))
    print("the data of test[1]:{}".format(data.test[1]))
    print("the data of test[780]:{}".format(data.test[780]))
    print("the type of test[780]:{}".format(type(data.test[780])))
    print("the shape of test[780][0]:{}".format(data.test[780][0].shape))


if __name__ == "__main__":
    test_SignFi("D:/CSI_Data/SignFi/")
