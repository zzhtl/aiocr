use burn::prelude::*;

use super::backbone::{CnnBackbone, CnnBackboneConfig};
use super::head::{CtcHead, CtcHeadConfig};

/// CRNN 文本识别模型
///
/// CNN 特征提取 + BiLSTM 序列建模 + CTC 解码
#[derive(Module, Debug)]
pub struct Crnn<B: Backend> {
    backbone: CnnBackbone<B>,
    head: CtcHead<B>,
}

/// CRNN 配置
#[derive(Config, Debug)]
pub struct CrnnConfig {
    /// 输入图片高度
    #[config(default = 32)]
    pub img_height: usize,
    /// 输入图片宽度
    #[config(default = 320)]
    pub img_width: usize,
    /// LSTM 隐藏层大小
    #[config(default = 256)]
    pub hidden_size: usize,
    /// 字符类别数（含 blank）
    pub num_classes: usize,
}

impl CrnnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Crnn<B> {
        let backbone = CnnBackboneConfig::new().init(device);
        let head = CtcHeadConfig::new(self.num_classes)
            .with_hidden_size(self.hidden_size)
            .init(device);

        Crnn { backbone, head }
    }
}

impl<B: Backend> Crnn<B> {
    /// 前向传播
    ///
    /// 输入: [batch, 1, H, W] 灰度图
    /// 输出: [batch, seq_len, num_classes] log probabilities
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        // CNN 特征提取: [batch, 512, 1, W']
        let features = self.backbone.forward(images);

        // 压缩高度维度: [batch, 512, W'] -> [batch, W', 512]
        let [batch, channels, _h, width] = features.dims();
        let features = features.reshape([batch, channels, width]).swap_dims(1, 2);

        // BiLSTM + CTC head: [batch, W', num_classes]
        self.head.forward(features)
    }
}
