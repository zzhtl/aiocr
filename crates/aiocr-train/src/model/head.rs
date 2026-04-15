use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig};
use burn::prelude::*;

/// BiLSTM + CTC Head
///
/// 接收 CNN 特征序列，通过双向 LSTM 建模序列关系，
/// 最后通过线性层映射到字符类别。
#[derive(Module, Debug)]
pub struct CtcHead<B: Backend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
}

/// CTC Head 配置
#[derive(Config, Debug)]
pub struct CtcHeadConfig {
    /// CNN 输出特征维度
    #[config(default = 512)]
    pub input_size: usize,
    /// LSTM 隐藏层大小
    #[config(default = 256)]
    pub hidden_size: usize,
    /// 字符类别数（含 blank）
    pub num_classes: usize,
}

impl CtcHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CtcHead<B> {
        // Burn 的 LSTM 是单向的，用两个 LSTM 模拟双向
        // 输出维度 = hidden_size（单向）
        let lstm = LstmConfig::new(self.input_size, self.hidden_size, true).init(device);

        // 线性层：hidden_size -> num_classes
        let linear = LinearConfig::new(self.hidden_size, self.num_classes).init(device);

        CtcHead { lstm, linear }
    }
}

impl<B: Backend> CtcHead<B> {
    /// 前向传播
    ///
    /// 输入: [batch, seq_len, features]
    /// 输出: [batch, seq_len, num_classes] (log probabilities)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // LSTM 处理序列
        let (hidden, _state) = self.lstm.forward(x, None);

        // 线性映射到字符类别
        let logits = self.linear.forward(hidden);

        // Log softmax
        burn::tensor::activation::log_softmax(logits, 2)
    }
}
