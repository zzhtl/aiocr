//! ONNX 模型推理后端
//!
//! 使用 tract-onnx 直接加载和推理 ONNX 格式模型文件，
//! 无需编译时转换，支持任意 PaddleOCR 兼容的检测/识别模型。

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use tract_onnx::pb;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::framework::Framework;

use crate::decode::CtcDecoder;
use crate::error::OcrError;
use crate::postprocess::{DbPostprocessConfig, db_postprocess};
use crate::preprocess::{preprocess_for_detection, preprocess_for_recognition};
use crate::types::BoundingBox;
use crate::{Detector, Recognizer};

type OcrPlan = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// 加载 ONNX 原始模型文件
fn load_onnx_proto(path: &Path) -> Result<pb::ModelProto, OcrError> {
    tract_onnx::onnx().proto_model_for_path(path).map_err(|e| {
        OcrError::ModelNotFound(format!("读取 ONNX 模型 {} 失败: {e}", path.display()))
    })
}

fn build_plan_from_proto(
    proto: &pb::ModelProto,
    input_shape: &[usize],
    model_name: &str,
) -> Result<OcrPlan, OcrError> {
    let patched = patch_model_input_shape(proto, input_shape)?;

    tract_onnx::onnx()
        .model_for_proto_model(&patched)
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .map_err(|e| OcrError::ModelNotFound(format!("加载 ONNX 模型 {model_name} 失败: {e}")))
}

fn patch_model_input_shape(
    proto: &pb::ModelProto,
    input_shape: &[usize],
) -> Result<pb::ModelProto, OcrError> {
    let mut patched = proto.clone();
    let graph = patched
        .graph
        .as_mut()
        .ok_or_else(|| OcrError::ModelNotFound("ONNX 模型缺少 graph".to_string()))?;

    let initializer_names = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<HashSet<_>>();

    let input_index = graph
        .input
        .iter()
        .position(|value| !initializer_names.contains(&value.name))
        .unwrap_or(0);

    if input_index >= graph.input.len() {
        return Err(OcrError::ModelNotFound(
            "ONNX 模型没有可用输入节点".to_string(),
        ));
    }

    let input_name = graph.input[input_index].name.clone();
    set_value_info_shape(&mut graph.input[input_index], input_shape)?;

    for (index, value) in graph.input.iter_mut().enumerate() {
        if index != input_index && contains_dynamic_dim(value) {
            clear_value_info_shape(value);
        }
    }
    for value in &mut graph.output {
        if contains_dynamic_dim(value) {
            clear_value_info_shape(value);
        }
    }
    for value in &mut graph.value_info {
        if value.name != input_name && contains_dynamic_dim(value) {
            clear_value_info_shape(value);
        }
    }

    Ok(patched)
}

fn set_value_info_shape(value: &mut pb::ValueInfoProto, shape: &[usize]) -> Result<(), OcrError> {
    let tensor = tensor_type_mut(value)?;
    let tensor_shape = tensor
        .shape
        .get_or_insert_with(|| pb::TensorShapeProto { dim: Vec::new() });
    tensor_shape.dim = shape
        .iter()
        .map(|dim| pb::tensor_shape_proto::Dimension {
            denotation: String::new(),
            value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(
                *dim as i64,
            )),
        })
        .collect();
    Ok(())
}

fn clear_value_info_shape(value: &mut pb::ValueInfoProto) {
    if let Some(tensor) = tensor_type_mut(value).ok() {
        tensor.shape = None;
    }
}

fn contains_dynamic_dim(value: &pb::ValueInfoProto) -> bool {
    let Some(r#type) = &value.r#type else {
        return false;
    };
    let Some(pb::type_proto::Value::TensorType(tensor)) = &r#type.value else {
        return false;
    };
    let Some(shape) = &tensor.shape else {
        return false;
    };
    shape.dim.iter().any(|dim| {
        matches!(
            dim.value,
            Some(pb::tensor_shape_proto::dimension::Value::DimParam(_))
        )
    })
}

fn tensor_type_mut(
    value: &mut pb::ValueInfoProto,
) -> Result<&mut pb::type_proto::Tensor, OcrError> {
    let r#type = value
        .r#type
        .as_mut()
        .ok_or_else(|| OcrError::ModelNotFound(format!("值 {} 缺少类型信息", value.name)))?;
    match r#type.value.as_mut() {
        Some(pb::type_proto::Value::TensorType(tensor)) => Ok(tensor),
        _ => Err(OcrError::ModelNotFound(format!(
            "值 {} 不是张量输入",
            value.name
        ))),
    }
}

/// 对给定输入执行推理，返回 (扁平化输出, 输出形状)
fn infer(
    plan: &OcrPlan,
    data: Vec<f32>,
    shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), OcrError> {
    let input = Tensor::from_shape(shape, &data)
        .map_err(|e| OcrError::Inference(format!("构建推理输入张量失败: {e}")))?;

    // 将 Tensor 转换为 TValue (Arc<Tensor>) 以匹配 plan.run() 接口
    let input_val: TValue = input.into();
    let outputs = plan
        .run(tvec![input_val])
        .map_err(|e| OcrError::Inference(format!("ONNX 推理失败: {e}")))?;

    let output_shape = outputs[0].shape().to_vec();
    let flat = outputs[0]
        .as_slice::<f32>()
        .map_err(|e| OcrError::Inference(format!("提取输出张量数据失败: {e}")))?
        .to_vec();

    Ok((flat, output_shape))
}

/// 基于 ONNX 运行时的文本检测器（DBNet/DBNet++）
pub struct OnnxDetector {
    proto: pb::ModelProto,
    plans: Mutex<HashMap<(usize, usize), OcrPlan>>,
    threshold: f32,
    box_threshold: f32,
    max_candidates: usize,
    unclip_ratio: f32,
}

impl OnnxDetector {
    /// 从 ONNX 模型文件创建检测器
    pub fn new(
        det_path: &Path,
        threshold: f32,
        box_threshold: f32,
        max_candidates: usize,
        unclip_ratio: f32,
    ) -> Result<Self, OcrError> {
        tracing::info!("加载 ONNX 检测模型: {}", det_path.display());
        Ok(Self {
            proto: load_onnx_proto(det_path)?,
            plans: Mutex::new(HashMap::new()),
            threshold,
            box_threshold,
            max_candidates,
            unclip_ratio,
        })
    }
}

impl Detector for OnnxDetector {
    fn detect(&self, img: &DynamicImage) -> Result<Vec<(BoundingBox, f32)>, OcrError> {
        let (input_data, meta) = preprocess_for_detection(img);
        let h = meta.resized_height as usize;
        let w = meta.resized_width as usize;

        let key = (h, w);
        let mut plans = self
            .plans
            .lock()
            .map_err(|err| OcrError::Inference(format!("锁定检测 ONNX 计划缓存失败: {err}")))?;
        if !plans.contains_key(&key) {
            let plan = build_plan_from_proto(&self.proto, &[1, 3, h, w], &format!("det[{h}x{w}]"))?;
            plans.insert(key, plan);
        }
        let plan = plans
            .get(&key)
            .ok_or_else(|| OcrError::Inference("检测 ONNX 计划缓存缺失".to_string()))?;

        let (flat, shape) = infer(plan, input_data, &[1, 3, h, w])?;

        // 检测模型输出形状可能为 [1,1,H,W]、[1,H,W] 或 [H,W]
        let (out_h, out_w) = extract_spatial_dims(&shape, h, w);
        let plane = out_h * out_w;

        if flat.len() < plane {
            return Err(OcrError::Inference(format!(
                "检测模型输出数据不足: 期望 {plane}, 实际 {}",
                flat.len()
            )));
        }

        let prob_map = if out_h != h || out_w != w {
            tracing::debug!("检测输出尺寸 {out_w}x{out_h} 与预处理 {w}x{h} 不符，执行重采样");
            resize_prob_map(&flat[..plane], out_w, out_h, w, h)
        } else {
            flat[..plane].to_vec()
        };

        let boxes = db_postprocess(
            &prob_map,
            h,
            w,
            DbPostprocessConfig {
                threshold: self.threshold,
                box_threshold: self.box_threshold,
                max_candidates: self.max_candidates,
                unclip_ratio: self.unclip_ratio,
                meta: &meta,
            },
        );

        Ok(boxes)
    }

    fn name(&self) -> &str {
        "onnx-det"
    }
}

/// 基于 ONNX 运行时的文本识别器（SVTR/CRNN+CTC）
pub struct OnnxRecognizer {
    proto: pb::ModelProto,
    plans: Mutex<HashMap<usize, OcrPlan>>,
    decoder: CtcDecoder,
    model_name: String,
}

impl OnnxRecognizer {
    /// 从 ONNX 模型文件创建识别器
    pub fn new(
        rec_path: &Path,
        decoder: CtcDecoder,
        model_name: impl Into<String>,
    ) -> Result<Self, OcrError> {
        tracing::info!("加载 ONNX 识别模型: {}", rec_path.display());
        Ok(Self {
            proto: load_onnx_proto(rec_path)?,
            plans: Mutex::new(HashMap::new()),
            decoder,
            model_name: model_name.into(),
        })
    }
}

impl Recognizer for OnnxRecognizer {
    fn recognize(&self, crop: &DynamicImage) -> Result<(String, f32), OcrError> {
        let input_data = preprocess_for_recognition(crop);
        let width = input_data.len() / (3 * 48);

        let mut plans = self
            .plans
            .lock()
            .map_err(|err| OcrError::Inference(format!("锁定识别 ONNX 计划缓存失败: {err}")))?;
        if !plans.contains_key(&width) {
            let plan = build_plan_from_proto(
                &self.proto,
                &[1, 3, 48, width],
                &format!("rec[48x{width}]"),
            )?;
            plans.insert(width, plan);
        }
        let plan = plans
            .get(&width)
            .ok_or_else(|| OcrError::Inference("识别 ONNX 计划缓存缺失".to_string()))?;

        let (flat, shape) = infer(plan, input_data, &[1, 3, 48, width])?;
        let num_classes = self.decoder.num_classes();

        // 兼容不同输出格式：[T,C]、[1,T,C]、[1,C,T]
        let (data, output_classes) = normalize_rec_output(flat, &shape, num_classes)?;
        Ok(self.decoder.decode_probabilities(&data, output_classes))
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

/// 从输出 shape 中提取空间维度 (H, W)
fn extract_spatial_dims(shape: &[usize], fallback_h: usize, fallback_w: usize) -> (usize, usize) {
    match shape.len() {
        4 => (shape[2], shape[3]), // [N, C, H, W]
        3 => (shape[1], shape[2]), // [N, H, W] 或 [C, H, W]
        2 => (shape[0], shape[1]), // [H, W]
        _ => (fallback_h, fallback_w),
    }
}

/// 将识别模型输出归一化为 [T*C] 扁平序列，返回 (data, output_classes)
fn normalize_rec_output(
    flat: Vec<f32>,
    shape: &[usize],
    dict_classes: usize,
) -> Result<(Vec<f32>, usize), OcrError> {
    // 剥离 batch 维，获得 rows x cols
    let (rows, cols) = match shape.len() {
        3 if shape[0] == 1 => (shape[1], shape[2]), // [1, T, C] 或 [1, C, T]
        2 => (shape[0], shape[1]),                  // [T, C] 或 [C, T]
        _ => {
            return Err(OcrError::Inference(format!(
                "识别模型输出形状不支持: {shape:?}"
            )));
        }
    };

    if cols >= rows || cols == dict_classes {
        // [T, C] 格式：cols 是类别数
        Ok((flat, cols))
    } else {
        // [C, T] 格式：rows 是类别数，需要转置
        let transposed = transpose_ct_to_tc(&flat, rows, cols);
        Ok((transposed, rows))
    }
}

/// 将 [C, T] 格式转置为 [T, C] 格式
fn transpose_ct_to_tc(values: &[f32], classes: usize, time_steps: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for c in 0..classes {
        for t in 0..time_steps {
            out[t * classes + c] = values[c * time_steps + t];
        }
    }
    out
}

/// 双线性插值缩放概率图
fn resize_prob_map(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut dst = vec![0.0f32; dst_w * dst_h];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        let sy = (y as f32 + 0.5) * scale_y - 0.5;
        let y0 = sy.floor().clamp(0.0, (src_h - 1) as f32) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);

        for x in 0..dst_w {
            let sx = (x as f32 + 0.5) * scale_x - 0.5;
            let x0 = sx.floor().clamp(0.0, (src_w - 1) as f32) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);

            let top = src[y0 * src_w + x0] * (1.0 - wx) + src[y0 * src_w + x1] * wx;
            let bot = src[y1 * src_w + x0] * (1.0 - wx) + src[y1 * src_w + x1] * wx;
            dst[y * dst_w + x] = top * (1.0 - wy) + bot * wy;
        }
    }
    dst
}
