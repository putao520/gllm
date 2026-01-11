use burn::backend::ndarray::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gllm::moe_layer::MoELayer;

fn bench_moe_layer(c: &mut Criterion) {
    let device = <NdArray<f32> as Backend>::Device::default();
    let hidden = 1024;
    let intermediate = 4096;
    let num_experts = 16;
    let top_k = 2;
    let batch = 4;
    let seq = 128;

    let moe = MoELayer::<NdArray<f32>>::new(
        &device,
        hidden,
        intermediate,
        num_experts,
        top_k,
        0,
    );
    let input = Tensor::<NdArray<f32>, 3>::random(
        [batch, seq, hidden],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );

    c.bench_function("moe_forward", |b| {
        b.iter(|| {
            let output = moe.forward(black_box(input.clone()));
            black_box(output);
        })
    });

}

criterion_group!(benches, bench_moe_layer);
criterion_main!(benches);
