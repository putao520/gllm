use burn::backend::ndarray::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor, TensorData};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use gllm::quantized::{GgmlDType, QTensor};
use gllm::quantized_ops::{CpuQuantizedBackend, MatmulInput, QuantizedBackend};
use half::f16;

fn bench_q4_matmul(c: &mut Criterion) {
    let device = <NdArray<f32> as Backend>::Device::default();
    let mut group = c.benchmark_group("q4_matmul");
    let cases = [(1, 1024, 1024), (4, 2048, 2048), (8, 4096, 4096)];

    for (batch, in_features, out_features) in cases {
        let input = Tensor::<NdArray<f32>, 2>::random(
            [batch, in_features],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let input_data = input
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("input data");
        let qtensor = build_q4_0_tensor(out_features, in_features);
        let (qweight, scales) = split_q4_0(&qtensor);
        let id = format!("{batch}x{in_features}x{out_features}");

        group.bench_with_input(BenchmarkId::new("naive", &id), &(), |b, _| {
            b.iter(|| {
                let weight_data = qtensor.dequantize();
                let weight = Tensor::from_data(
                    TensorData::new(weight_data, [out_features, in_features]),
                    &device,
                )
                .transpose();
                let output = input.clone().matmul(weight);
                black_box(output);
            })
        });

        group.bench_with_input(BenchmarkId::new("optimized", &id), &(), |b, _| {
            b.iter(|| {
                let output_data = CpuQuantizedBackend::q4_matmul(
                    MatmulInput::new(&input_data, batch, in_features),
                    &qweight,
                    &scales,
                );
                let output = Tensor::from_data(
                    TensorData::new(output_data, [batch, out_features]),
                    &device,
                );
                black_box(output);
            })
        });
    }

    group.finish();
}

fn build_q4_0_tensor(out_features: usize, in_features: usize) -> QTensor {
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let blocks_per_row = in_features / BLOCK;
    let total_blocks = out_features * blocks_per_row;
    let mut data = Vec::with_capacity(total_blocks * BLOCK_BYTES);

    for block_idx in 0..total_blocks {
        let scale = f16::from_f32(0.02 + (block_idx % 7) as f32 * 0.001);
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        for i in 0..16u8 {
            let byte = ((block_idx as u8) & 0xF) | ((i & 0xF) << 4);
            data.push(byte);
        }
    }

    QTensor {
        data,
        dtype: GgmlDType::Q4_0,
        shape: vec![out_features, in_features],
    }
}

fn split_q4_0(qtensor: &QTensor) -> (Vec<u8>, Vec<f16>) {
    const BLOCK_BYTES: usize = 18;
    const QBYTES: usize = 16;
    let blocks = qtensor.data.len() / BLOCK_BYTES;
    let mut scales = Vec::with_capacity(blocks);
    let mut qweight = Vec::with_capacity(blocks * QBYTES);

    for block in qtensor.data.chunks_exact(BLOCK_BYTES) {
        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]]));
        scales.push(scale);
        qweight.extend_from_slice(&block[2..]);
    }

    (qweight, scales)
}

criterion_group!(benches, bench_q4_matmul);
criterion_main!(benches);
