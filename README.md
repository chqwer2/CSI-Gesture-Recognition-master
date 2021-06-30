# CSI-Gesture-Recognition-master
=============

This code was using for increasingly popular gesture recognition trained on CSI（Channel State Information,信道状态信息） data. 

Our method is based on [Deep Metric Learning](https://github.com/bnu-wangxun/Deep_Metric) and choosing CNN/ [ResNet](https://arxiv.org/abs/1512.03385 )/ [VGG ](https://arxiv.org/pdf/1409.1556.pdf) as backbone network. We conduct and evaluate this three types of network with different structure on [SignFi Dataset](https://dl.acm.org/doi/pdf/10.1145/3191755).  However, for better results, we make some augmentation on the SignFi Dataset and shared is [here](https://pan.baidu.com/s/1Khs6PIY1tp5QSZpdFmbW7), the password is "42Wy".

We call the result Network structure **CSI-Gesture-Recognition-Network**.



## Future Work

2020/6/27

This work is done for *IoT lesson on 2020 spring term, Sun Yat-sen University*.  For further improvement, we can implement SVMs or Softmax Function as the state-of-the-art classifier to tackle the Feature Map output by our proposed **CSI-Gesture-Recognition-Network**.
