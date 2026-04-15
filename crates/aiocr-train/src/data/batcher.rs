use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use super::dataset::OcrItem;

/// OCR 批处理器
#[derive(Clone)]
pub struct OcrBatcher {
    img_height: usize,
    img_width: usize,
}

/// OCR 批数据
#[derive(Debug, Clone)]
pub struct OcrBatch<B: Backend> {
    /// 图片张量 [batch, 1, H, W]
    pub images: Tensor<B, 4>,
    /// 标签索引 [batch, max_label_len]（padding 用 0）
    pub targets: Tensor<B, 2, Int>,
    /// 每个样本的标签长度 [batch]
    pub target_lengths: Tensor<B, 1, Int>,
}

impl OcrBatcher {
    pub fn new(img_height: usize, img_width: usize) -> Self {
        Self {
            img_height,
            img_width,
        }
    }
}

impl<B: Backend> Batcher<B, OcrItem, OcrBatch<B>> for OcrBatcher {
    fn batch(&self, items: Vec<OcrItem>, device: &B::Device) -> OcrBatch<B> {
        let batch_size = items.len();

        // 加载并预处理图片
        let mut image_data = Vec::with_capacity(batch_size * self.img_height * self.img_width);
        let mut max_label_len = 0;

        for item in &items {
            max_label_len = max_label_len.max(item.label_indices.len());

            // 加载图片并转为灰度
            let img = match image::open(&item.image_path) {
                Ok(img) => img.to_luma8(),
                Err(e) => {
                    tracing::warn!("加载图片失败 {:?}: {e}", item.image_path);
                    // 用黑色图片填充
                    image::GrayImage::new(self.img_width as u32, self.img_height as u32)
                }
            };

            // 缩放到目标尺寸
            let resized = image::imageops::resize(
                &img,
                self.img_width as u32,
                self.img_height as u32,
                image::imageops::FilterType::Lanczos3,
            );

            // 归一化到 [-1, 1]
            for pixel in resized.as_raw() {
                image_data.push(*pixel as f32 / 255.0 * 2.0 - 1.0);
            }
        }

        // 构建图片张量 [batch, 1, H, W]
        let images = Tensor::<B, 1>::from_floats(image_data.as_slice(), device).reshape([
            batch_size,
            1,
            self.img_height,
            self.img_width,
        ]);

        // 构建标签张量（padding 到最大长度）
        let mut target_data = vec![0i64; batch_size * max_label_len];
        let mut length_data = Vec::with_capacity(batch_size);

        for (i, item) in items.iter().enumerate() {
            length_data.push(item.label_indices.len() as i64);
            for (j, &idx) in item.label_indices.iter().enumerate() {
                target_data[i * max_label_len + j] = idx as i64;
            }
        }

        let targets = Tensor::<B, 1, Int>::from_ints(target_data.as_slice(), device)
            .reshape([batch_size, max_label_len]);

        let target_lengths = Tensor::<B, 1, Int>::from_ints(length_data.as_slice(), device);

        OcrBatch {
            images,
            targets,
            target_lengths,
        }
    }
}
