# AIOCR

纯 Rust OCR 桌面应用 workspace，包含：

- `aiocr-core`: OCR 推理管线、图片预处理、后处理、可插拔检测/识别 trait
- `aiocr-train`: 基于当前本地 AI 识别模型继续训练、模型导出与热切换
- `aiocr-gui`: `eframe + egui` 桌面界面

## 当前状态

已经打通的主路径：

- GUI 选择图片
- GUI 拖拽图片到窗口
- GUI 支持 `Ctrl+V / Cmd+V` 直接粘贴截图并开始识别
- 默认优先加载 `models/` 下的 PP-OCRv5 Server 模型，检测优先走 ONNX，识别优先走 Burn
- GUI 选择数据集目录并继续训练当前本地 AI 识别模型
- 训练产物保存到 `artifacts/`
- GUI 列出训练后的本地 AI 模型并切换激活识别器
- 激活模型后执行 OCR

当前仓库内的默认 OCR 主路径是：

1. `aiocr-core` 默认优先使用 `models/` 下的 PP-OCRv5 Server 模型：检测优先走 ONNX、识别优先走 Burn，方向分类继续沿用现有 `cls`
2. `aiocr-train` 会基于当前激活的识别 AI 模型继续训练识别权重，并导出新的本地模型包
3. GUI 通过 `Recognizer` trait 在默认本地 PaddleOCR 识别器和训练后 AI 模型之间切换

## ONNX / Burn 说明

仓库已经包含 `burn-onnx` 的 `build.rs` 转换脚手架，以及已经下载好的模型文件：

- `models/det.onnx`
- `models/cls.onnx`
- `models/rec.onnx`
- `models/ppocr_keys_v1.txt`

当前构建脚本会在编译时：

- 检测到兼容的 `cls.onnx` 与经静态化处理后可转换的 `det.onnx / rec.onnx` 后生成对应 Burn 模块
- 使用 `Embedded` 策略把 `.bpk` 嵌入二进制
- 对 `burn-onnx 0.21.0-pre.3` 生成的 OCR 代码做兼容补丁，修正 shape 拼接代码
- 通过编译期 `cfg` 暴露生成模块可用性

运行时默认走 `models/` 目录中的 OCR 模型：

- 检测：PP-OCRv5 Server det（优先 ONNX，缺失时回退 Burn）
- 方向分类：现有 2-class cls
- 识别：PP-OCRv5 Server rec（优先 Burn，失败回退 ONNX）

如果需要更强的基础模型，可替换 `models/det.onnx`、`models/rec.onnx` 和对应字典后重新构建；GUI 训练得到的新模型会以外部 `rec.bpk` 权重包形式保存在 `artifacts/` 下。

## 数据集格式

```text
dataset/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── labels.txt
```

`labels.txt` 格式：

```text
0001.png	示例文本
0002.png	another text
```

建议先直接体验默认 PP-OCRv5 Server 推理；如果场景里有大量专有字体、票据格式或特定手写体，再基于当前本地 AI 模型继续训练做针对性提升。

## 运行

检查构建：

```bash
cargo check
```

运行 GUI（实际使用强烈建议 `--release`，否则本地 OCR 推理仍会明显偏慢）：

```bash
cargo run --release -p aiocr-gui
```

运行测试：

```bash
cargo test
```

## GUI 使用流程

默认 OCR：

1. 打开 `OCR 识别` 标签页
2. 载入图片，或直接按 `Ctrl+V / Cmd+V` 粘贴截图
3. 点击识别，默认走 `models/` 目录中的 PP-OCRv5 Server 模型

界面会在启动时自动尝试加载系统中文字体，优先兼容 Linux / macOS / Windows 的常见中文字体。

微调 / 模型切换：

1. 打开 `Training` 标签页
2. 选择数据集目录
3. 选择训练产物目录
4. 确认当前激活的是你要继续训练的本地 AI 模型
5. 调整 `Epoch / Batch / 学习率`
6. 点击 `开始训练`
7. 训练完成后会自动切换到新模型
8. 返回 `OCR 识别` 标签页，重新识别图片并对比效果

更换更强基础模型：

1. 准备新的 `det / rec` ONNX 和对应字典；`cls` 可选保留现有文件
2. 替换 `models/` 下的同名文件
3. 重新执行 `cargo check` 或重新构建 GUI
4. 重新打开 GUI，此时默认直接识别会使用新的本地 AI 模型

## 目录结构

```text
aiocr/
├── models/
├── crates/
│   ├── aiocr-core/
│   ├── aiocr-train/
│   └── aiocr-gui/
└── Cargo.toml
```

## 验证范围

当前测试覆盖：

- `aiocr-core` 的 CTC 解码、预处理、后处理、后备检测
- `aiocr-core` 的真实 `det / cls / rec` 模型 smoke test
- `aiocr-train` 的 AI 模型清单、模型导出、模型加载

仍建议后续继续补：

- 端到端 GUI smoke test
- 包含中英文混排样本的集成测试
- 真实票据/截图/手写样本上的精度基准
