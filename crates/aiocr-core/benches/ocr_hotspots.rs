use aiocr_core::build_spatial_text;
use aiocr_core::config::{DetectionBoxMode, DetectionResize};
use aiocr_core::postprocess::{DbPostprocessConfig, db_postprocess};
use aiocr_core::preprocess::{preprocess_for_detection, preprocess_for_recognition};
use aiocr_core::types::{BoundingBox, ImageMeta, TextDirection, TextRegion};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use image::{DynamicImage, Rgb, RgbImage};

fn bench_detection_preprocess(c: &mut Criterion) {
    let image = sample_document_image(1600, 1000);

    c.bench_function("preprocess_for_detection_1600x1000", |b| {
        b.iter(|| {
            let output =
                preprocess_for_detection(black_box(&image), DetectionResize::limit_side(1600, false));
            black_box(output);
        });
    });
}

fn bench_recognition_preprocess(c: &mut Criterion) {
    let crop = sample_document_image(640, 48);

    c.bench_function("preprocess_for_recognition_640x48", |b| {
        b.iter(|| {
            let output = preprocess_for_recognition(black_box(&crop));
            black_box(output);
        });
    });
}

fn bench_db_postprocess(c: &mut Criterion) {
    let width = 512usize;
    let height = 512usize;
    let prob_map = sample_probability_map(width, height);
    let meta = ImageMeta {
        orig_width: width as u32,
        orig_height: height as u32,
        resized_width: width as u32,
        resized_height: height as u32,
        content_width: width as u32,
        content_height: height as u32,
        pad_x: 0,
        pad_y: 0,
        scale_x: 1.0,
        scale_y: 1.0,
    };
    let config = DbPostprocessConfig {
        threshold: 0.5,
        box_threshold: 0.6,
        max_candidates: 256,
        unclip_ratio: 1.2,
        min_box_area: 4.0,
        box_mode: DetectionBoxMode::AxisAligned,
        meta: &meta,
    };

    c.bench_function("db_postprocess_512_many_components", |b| {
        b.iter(|| {
            let boxes = db_postprocess(black_box(&prob_map), height, width, black_box(config));
            black_box(boxes);
        });
    });
}

fn bench_spatial_text(c: &mut Criterion) {
    let regions = sample_text_regions(240);

    c.bench_function("build_spatial_text_240_regions", |b| {
        b.iter(|| {
            let text = build_spatial_text(black_box(&regions), 1600.0, 1000.0);
            black_box(text);
        });
    });
}

fn sample_document_image(width: u32, height: u32) -> DynamicImage {
    let mut image = RgbImage::from_pixel(width, height, Rgb([255, 255, 255]));

    for row in 0..24u32 {
        let y = 24 + row * 36;
        if y + 12 >= height {
            break;
        }

        for x in 32..width.saturating_sub(32) {
            if x % 17 < 12 {
                for dy in 0..10 {
                    image.put_pixel(x, y + dy, Rgb([20, 20, 20]));
                }
            }
        }
    }

    DynamicImage::ImageRgb8(image)
}

fn sample_probability_map(width: usize, height: usize) -> Vec<f32> {
    let mut map = vec![0.02f32; width * height];

    for row in 0..32usize {
        let y = 8 + row * 15;
        for col in 0..6usize {
            let x = 12 + col * 78 + row % 3;
            fill_rect(&mut map, width, height, x, y, 48, 6, 0.92);
        }
    }

    for y in (3..height).step_by(19) {
        for x in (5..width).step_by(23) {
            map[y * width + x] = 0.55;
        }
    }

    map
}

fn fill_rect(
    map: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    rect_w: usize,
    rect_h: usize,
    value: f32,
) {
    for yy in y..(y + rect_h).min(height) {
        for xx in x..(x + rect_w).min(width) {
            map[yy * width + xx] = value;
        }
    }
}

fn sample_text_regions(count: usize) -> Vec<TextRegion> {
    (0..count)
        .map(|index| {
            let row = index / 6;
            let col = index % 6;
            let x = 32.0 + col as f32 * 240.0;
            let y = 24.0 + row as f32 * 24.0;
            TextRegion {
                bbox: BoundingBox::from_rect(x, y, 120.0, 16.0),
                confidence: 0.9,
                text: format!("text-{index}"),
                direction: TextDirection::Horizontal,
            }
        })
        .collect()
}

criterion_group!(
    benches,
    bench_detection_preprocess,
    bench_recognition_preprocess,
    bench_db_postprocess,
    bench_spatial_text
);
criterion_main!(benches);
