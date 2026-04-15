use burn::nn::{
    BatchNorm, BatchNormConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
};
use burn::prelude::*;

/// CNN 特征提取骨干网络
///
/// 简化版 ResNet 风格 CNN，输出 [batch, 512, 1, W/4] 的特征图
#[derive(Module, Debug)]
pub struct CnnBackbone<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    pool1: MaxPool2d,

    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    pool2: MaxPool2d,

    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,

    conv4: Conv2d<B>,
    bn4: BatchNorm<B>,
    pool3: MaxPool2d,

    conv5: Conv2d<B>,
    bn5: BatchNorm<B>,

    conv6: Conv2d<B>,
    bn6: BatchNorm<B>,
    pool4: MaxPool2d,
}

/// CNN 骨干网络配置
#[derive(Config, Debug)]
pub struct CnnBackboneConfig {
    #[config(default = 1)]
    pub in_channels: usize,
}

impl CnnBackboneConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CnnBackbone<B> {
        let pad3 = PaddingConfig2d::Explicit(1, 1, 1, 1);

        CnnBackbone {
            conv1: Conv2dConfig::new([self.in_channels, 64], [3, 3])
                .with_padding(pad3.clone())
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),

            conv2: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(pad3.clone())
                .init(device),
            bn2: BatchNormConfig::new(128).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),

            conv3: Conv2dConfig::new([128, 256], [3, 3])
                .with_padding(pad3.clone())
                .init(device),
            bn3: BatchNormConfig::new(256).init(device),

            conv4: Conv2dConfig::new([256, 256], [3, 3])
                .with_padding(pad3.clone())
                .init(device),
            bn4: BatchNormConfig::new(256).init(device),
            pool3: MaxPool2dConfig::new([2, 1]).with_strides([2, 1]).init(),

            conv5: Conv2dConfig::new([256, 512], [3, 3])
                .with_padding(pad3.clone())
                .init(device),
            bn5: BatchNormConfig::new(512).init(device),

            conv6: Conv2dConfig::new([512, 512], [3, 3])
                .with_padding(pad3)
                .init(device),
            bn6: BatchNormConfig::new(512).init(device),
            pool4: MaxPool2dConfig::new([2, 1]).with_strides([2, 1]).init(),
        }
    }
}

impl<B: Backend> CnnBackbone<B> {
    /// 前向传播
    ///
    /// 输入: [batch, 1, H, W] 灰度图
    /// 输出: [batch, 512, 1, W'] 特征图
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.pool1.forward(burn::tensor::activation::relu(
            self.bn1.forward(self.conv1.forward(x)),
        ));
        let x = self.pool2.forward(burn::tensor::activation::relu(
            self.bn2.forward(self.conv2.forward(x)),
        ));
        let x = burn::tensor::activation::relu(self.bn3.forward(self.conv3.forward(x)));
        let x = self.pool3.forward(burn::tensor::activation::relu(
            self.bn4.forward(self.conv4.forward(x)),
        ));
        let x = burn::tensor::activation::relu(self.bn5.forward(self.conv5.forward(x)));
        self.pool4.forward(burn::tensor::activation::relu(
            self.bn6.forward(self.conv6.forward(x)),
        ))
    }
}
