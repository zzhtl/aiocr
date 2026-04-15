use burn_onnx::{LoadStrategy, ModelGen};
use prost::Message;
use std::path::PathBuf;
use tract_onnx::pb;
use tract_onnx::tract_core::framework::Framework;

const REC_BURN_INPUT_SHAPE: [usize; 4] = [1, 3, 48, 320];

fn main() {
    let rec_path = "../../models/rec.onnx";
    println!("cargo:rerun-if-changed={rec_path}");
    println!("cargo:rustc-check-cfg=cfg(aiocr_has_generated_rec)");

    if std::path::Path::new(rec_path).exists() && can_generate_rec(rec_path) {
        println!("cargo:rustc-cfg=aiocr_has_generated_rec");
    }
}

fn can_generate_rec(rec_path: &str) -> bool {
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set")).join("probe-rec");
    let out_dir_str = out_dir.to_string_lossy().into_owned();

    let result = std::panic::catch_unwind(|| {
        let patched = patch_model_input_shape(rec_path, &REC_BURN_INPUT_SHAPE);
        ModelGen::new()
            .input(
                patched
                    .to_str()
                    .expect("patched rec path is not valid UTF-8"),
            )
            .out_dir(&out_dir_str)
            .load_strategy(LoadStrategy::Embedded)
            .run_from_script();
    });

    let _ = std::fs::remove_dir_all(&out_dir);

    result.is_ok()
}

fn patch_model_input_shape(input: &str, shape: &[usize]) -> PathBuf {
    let mut proto = tract_onnx::onnx()
        .proto_model_for_path(input)
        .unwrap_or_else(|err| panic!("failed to read ONNX model {input}: {err}"));
    let graph = proto
        .graph
        .as_mut()
        .unwrap_or_else(|| panic!("ONNX model {input} has no graph"));

    let initializer_names = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<std::collections::HashSet<_>>();

    let input_index = graph
        .input
        .iter()
        .position(|value| !initializer_names.contains(&value.name))
        .unwrap_or(0);

    if input_index >= graph.input.len() {
        panic!("ONNX model {input} has no usable graph input");
    }

    let input_name = graph.input[input_index].name.clone();
    set_value_info_shape(&mut graph.input[input_index], shape);

    for (index, value) in graph.input.iter_mut().enumerate() {
        if index != input_index && contains_dynamic_dim(value) {
            clear_value_info_shape(value);
        }
    }
    for value in graph.output.iter_mut() {
        if contains_dynamic_dim(value) {
            clear_value_info_shape(value);
        }
    }
    for value in graph.value_info.iter_mut() {
        if value.name != input_name && contains_dynamic_dim(value) {
            clear_value_info_shape(value);
        }
    }

    let patched =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set")).join("patched-rec.onnx");
    let bytes = proto.encode_to_vec();
    std::fs::write(&patched, bytes).unwrap_or_else(|err| {
        panic!(
            "failed to write patched ONNX model {}: {err}",
            patched.display()
        )
    });
    patched
}

fn set_value_info_shape(value: &mut pb::ValueInfoProto, shape: &[usize]) {
    let tensor = tensor_type_mut(value);
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
}

fn clear_value_info_shape(value: &mut pb::ValueInfoProto) {
    if let Some(tensor) = tensor_type_mut_opt(value) {
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

fn tensor_type_mut(value: &mut pb::ValueInfoProto) -> &mut pb::type_proto::Tensor {
    let name = value.name.clone();
    tensor_type_mut_opt(value).unwrap_or_else(|| panic!("value {name} is not a tensor input"))
}

fn tensor_type_mut_opt(value: &mut pb::ValueInfoProto) -> Option<&mut pb::type_proto::Tensor> {
    let r#type = value.r#type.as_mut()?;
    match r#type.value.as_mut() {
        Some(pb::type_proto::Value::TensorType(tensor)) => Some(tensor),
        _ => None,
    }
}
