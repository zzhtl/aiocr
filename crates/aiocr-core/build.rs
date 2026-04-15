use burn_onnx::{LoadStrategy, ModelGen};
use prost::Message;
use std::fs;
use std::path::{Path, PathBuf};
use tract_onnx::pb;
use tract_onnx::tract_core::framework::Framework;

const DET_BURN_INPUT_SHAPE: [usize; 4] = [1, 3, 512, 512];
const REC_BURN_INPUT_SHAPE: [usize; 4] = [1, 3, 48, 320];

fn main() {
    // 编译时将 ONNX 模型转换为纯 Burn Rust 代码
    // 模型文件需要先下载到 models/ 目录
    let models_dir = "../../models";
    println!("cargo:rerun-if-changed={models_dir}");
    println!("cargo:rustc-check-cfg=cfg(aiocr_has_det)");
    println!("cargo:rustc-check-cfg=cfg(aiocr_has_cls)");
    println!("cargo:rustc-check-cfg=cfg(aiocr_has_rec)");

    let det_path = format!("{models_dir}/det.onnx");
    println!("cargo:rerun-if-changed={det_path}");
    if std::path::Path::new(&det_path).exists() {
        try_generate_model(
            &det_path,
            "models/det/",
            "det",
            "aiocr_has_det",
            Some(&DET_BURN_INPUT_SHAPE),
        );
    }

    let cls_path = format!("{models_dir}/cls.onnx");
    println!("cargo:rerun-if-changed={cls_path}");
    if std::path::Path::new(&cls_path).exists() {
        try_generate_model(&cls_path, "models/cls/", "cls", "aiocr_has_cls", None);
    }

    let rec_path = format!("{models_dir}/rec.onnx");
    println!("cargo:rerun-if-changed={rec_path}");
    if std::path::Path::new(&rec_path).exists() {
        try_generate_model(
            &rec_path,
            "models/rec/",
            "rec",
            "aiocr_has_rec",
            Some(&REC_BURN_INPUT_SHAPE),
        );
    }
}

fn try_generate_model(
    input: &str,
    out_dir: &str,
    stem: &str,
    cfg_name: &str,
    static_input_shape: Option<&[usize]>,
) {
    match std::panic::catch_unwind(|| generate_model(input, out_dir, stem, static_input_shape)) {
        Ok(()) => println!("cargo:rustc-cfg={cfg_name}"),
        Err(err) => {
            println!(
                "cargo:warning=skip burn-onnx generation for {input}: {}",
                panic_message(&err)
            );
        }
    }
}

fn generate_model(input: &str, out_dir: &str, stem: &str, static_input_shape: Option<&[usize]>) {
    let patched_input;
    let input = if let Some(shape) = static_input_shape {
        patched_input = patch_model_input_shape(input, shape);
        patched_input
            .as_path()
            .to_str()
            .expect("patched input path is not valid UTF-8")
    } else {
        input
    };

    ModelGen::new()
        .input(input)
        .out_dir(out_dir)
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();

    let generated = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"))
        .join(out_dir)
        .join(format!("{stem}.rs"));
    patch_generated_source(&generated);
}

fn patch_generated_source(path: &Path) {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read generated model {}: {err}", path.display()));
    let (mut source, cast_patched) = patch_cast_shape_fragments(&source);

    if cast_patched && !source.contains("fn tensor1_to_i64_vec") {
        source = source.replacen(
            "use burn_store::ModuleSnapshot;\n\n",
            "use burn_store::ModuleSnapshot;\n\nfn tensor1_to_i64_vec<B: Backend>(tensor: &Tensor<B, 1, Int>) -> Vec<i64> {\n    tensor\n        .to_data()\n        .to_vec::<i64>()\n        .expect(\"Failed to materialize 1D int tensor shape fragment\")\n}\n\n",
            1,
        );
    }

    fs::write(path, source)
        .unwrap_or_else(|err| panic!("failed to write generated model {}: {err}", path.display()));
}

fn patch_cast_shape_fragments(source: &str) -> (String, bool) {
    let mut output = String::with_capacity(source.len());
    let mut changed = false;

    for line in source.split_inclusive('\n') {
        let mut patched = line.to_owned();
        loop {
            let Some(start) = patched.find("&cast") else {
                break;
            };
            let suffix = &patched[start + 1..];
            let Some(bracket_pos) = suffix.find("[..]") else {
                break;
            };
            let var_name = &suffix[..bracket_pos];
            if !var_name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            {
                break;
            }

            let replace_end = start + 1 + bracket_pos + 4;
            let replacement = format!("tensor1_to_i64_vec(&{var_name}).as_slice()");
            patched.replace_range(start..replace_end, &replacement);
            changed = true;
        }
        output.push_str(&patched);
    }

    (output, changed)
}

fn panic_message(err: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = err.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = err.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
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

    let input_path = Path::new(input);
    let patched_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"))
        .join("patched-model-inputs");
    fs::create_dir_all(&patched_dir).unwrap_or_else(|err| {
        panic!(
            "failed to create patched ONNX dir {}: {err}",
            patched_dir.display()
        )
    });
    let patched_name = input_path
        .file_name()
        .unwrap_or_else(|| panic!("failed to get file name from ONNX path {input}"));
    let patched = patched_dir.join(patched_name);
    let bytes = proto.encode_to_vec();
    fs::write(&patched, bytes).unwrap_or_else(|err| {
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
