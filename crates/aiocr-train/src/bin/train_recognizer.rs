use std::collections::HashMap;
use std::path::PathBuf;

use aiocr_train::{
    Trainer, TrainingCallback, TrainingConfig, TrainingProgress, TrainingSummary,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_config()?;
    let mut callback = StdoutCallback;
    let summary = Trainer::new(config).train(&mut callback)?;
    println!("model_dir={}", summary.model_dir.display());
    Ok(())
}

fn parse_config() -> Result<TrainingConfig, String> {
    let values = parse_key_values(std::env::args().skip(1).collect());
    let mut config = TrainingConfig::default();

    if let Some(dataset) = values.get("dataset") {
        config.dataset_path = PathBuf::from(dataset);
    }
    if let Some(artifacts) = values.get("artifacts") {
        config.artifact_dir = PathBuf::from(artifacts);
    }
    if let Some(base_model) = values.get("base-model") {
        config.base_model_dir = Some(PathBuf::from(base_model));
    }
    if let Some(epochs) = values.get("epochs") {
        config.num_epochs = parse_value("epochs", epochs)?;
    }
    if let Some(batch) = values.get("batch") {
        config.batch_size = parse_value("batch", batch)?;
    }
    if let Some(lr) = values.get("lr") {
        config.learning_rate = parse_value("lr", lr)?;
    }

    Ok(config)
}

fn parse_key_values(args: Vec<String>) -> HashMap<String, String> {
    let mut values = HashMap::new();
    let mut index = 0usize;
    while index < args.len() {
        let arg = &args[index];
        if let Some(key) = arg.strip_prefix("--") {
            if let Some((key, value)) = key.split_once('=') {
                values.insert(key.to_string(), value.to_string());
            } else if let Some(value) = args.get(index + 1) {
                values.insert(key.to_string(), value.clone());
                index += 1;
            }
        }
        index += 1;
    }
    values
}

fn parse_value<T>(key: &str, value: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|err| format!("参数 --{key}={value} 无效: {err}"))
}

struct StdoutCallback;

impl TrainingCallback for StdoutCallback {
    fn on_epoch_start(&mut self, epoch: usize, total_epochs: usize) {
        println!("epoch_start {epoch}/{total_epochs}");
    }

    fn on_batch_end(&mut self, progress: &TrainingProgress) {
        println!(
            "batch epoch={}/{} batch={}/{} loss={:.4} acc={:.4}",
            progress.epoch,
            progress.total_epochs,
            progress.batch,
            progress.total_batches,
            progress.loss,
            progress.accuracy,
        );
    }

    fn on_epoch_end(&mut self, progress: &TrainingProgress) {
        println!(
            "epoch_end epoch={}/{} loss={:.4} val_acc={:.4}",
            progress.epoch, progress.total_epochs, progress.loss, progress.accuracy,
        );
    }

    fn on_training_complete(&mut self, summary: &TrainingSummary) {
        println!(
            "training_complete model={} accuracy={:.4} loss={:.4}",
            summary.model.display_name(),
            summary.model.accuracy,
            summary.model.avg_loss,
        );
    }
}
