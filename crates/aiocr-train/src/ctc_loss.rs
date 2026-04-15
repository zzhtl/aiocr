use burn::prelude::*;

/// CTC 损失函数
///
/// 简化实现：使用负对数似然近似 CTC loss
/// 完整的 CTC 前向-后向算法较复杂，这里先用简化版本
#[allow(clippy::single_range_in_vec_init)]
pub fn ctc_loss<B: Backend>(
    log_probs: Tensor<B, 3>,           // [batch, time, classes]
    targets: Tensor<B, 2, Int>,        // [batch, max_target_len]
    target_lengths: Tensor<B, 1, Int>, // [batch]
) -> Tensor<B, 1> {
    let [batch_size, time_steps, _num_classes] = log_probs.dims();

    // 简化 CTC: 对每个时间步取目标字符的负对数概率
    // 这不是真正的 CTC loss，但作为初始实现可以驱动训练
    let mut losses = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let sample_log_probs = log_probs.clone().slice([b..b + 1]);
        let sample_log_probs: Tensor<B, 2> = sample_log_probs.squeeze_dim(0);

        let sample_targets = targets.clone().slice([b..b + 1]);
        let sample_targets: Tensor<B, 1, Int> = sample_targets.squeeze_dim(0);

        // 获取目标长度
        let tgt_len_data = target_lengths.clone().slice([b..b + 1]);
        let tgt_len = tgt_len_data.into_scalar().elem::<i64>() as usize;

        if tgt_len == 0 || time_steps == 0 {
            losses.push(0.0f32);
            continue;
        }

        // 均匀分配时间步到目标字符
        let step = time_steps as f32 / tgt_len as f32;
        let mut loss_sum = 0.0f32;

        for t_idx in 0..tgt_len {
            let time_pos = (t_idx as f32 * step) as usize;
            let time_pos = time_pos.min(time_steps - 1);

            let target_class = sample_targets
                .clone()
                .slice([t_idx..t_idx + 1])
                .into_scalar()
                .elem::<i64>() as usize;

            let log_prob = sample_log_probs
                .clone()
                .slice([time_pos..time_pos + 1, target_class..target_class + 1])
                .into_scalar()
                .elem::<f32>();

            loss_sum -= log_prob;
        }

        losses.push(loss_sum / tgt_len as f32);
    }

    let device = log_probs.device();
    Tensor::<B, 1>::from_floats(losses.as_slice(), &device)
}
