//! 诊断: 同一 Client 连续 embed 是否产生一致结果。

use gllm::Client;

#[test]
#[ignore = "诊断用,手动跑"]
fn debug_embed_determinism() {
    let client = Client::new_embedding("intfloat/e5-small-v2").expect("load");

    let r1 = client.embed(["hello"]).expect("embed1");
    let r2 = client.embed(["hello"]).expect("embed2");
    let r3 = client.embed(["hello"]).expect("embed3");

    let e1 = &r1.embeddings[0].embedding;
    let e2 = &r2.embeddings[0].embedding;
    let e3 = &r3.embeddings[0].embedding;

    eprintln!("e1[..8] = {:?}", &e1[..8]);
    eprintln!("e2[..8] = {:?}", &e2[..8]);
    eprintln!("e3[..8] = {:?}", &e3[..8]);

    let same_12: usize = e1.iter().zip(e2.iter()).filter(|(a, b)| (*a - *b).abs() < 1e-5).count();
    let same_23: usize = e2.iter().zip(e3.iter()).filter(|(a, b)| (*a - *b).abs() < 1e-5).count();
    eprintln!("same(e1,e2) {}/{}, same(e2,e3) {}/{}", same_12, e1.len(), same_23, e2.len());

    assert_eq!(e1, e2, "embed 1st and 2nd must produce identical output");
    assert_eq!(e2, e3, "embed 2nd and 3rd must produce identical output");
}

#[test]
#[ignore = "诊断用,手动跑 — 测试不同输入两次 embed 是否各自一致"]
fn debug_embed_determinism_diff_inputs() {
    let client = Client::new_embedding("intfloat/e5-small-v2").expect("load");

    let a1 = client.embed(["alpha"]).expect("alpha1");
    let b1 = client.embed(["beta"]).expect("beta1");
    let a2 = client.embed(["alpha"]).expect("alpha2");
    let b2 = client.embed(["beta"]).expect("beta2");

    let ea1 = &a1.embeddings[0].embedding;
    let ea2 = &a2.embeddings[0].embedding;
    let eb1 = &b1.embeddings[0].embedding;
    let eb2 = &b2.embeddings[0].embedding;

    eprintln!("alpha1[..4] = {:?}", &ea1[..4]);
    eprintln!("alpha2[..4] = {:?}", &ea2[..4]);
    eprintln!("beta1[..4]  = {:?}", &eb1[..4]);
    eprintln!("beta2[..4]  = {:?}", &eb2[..4]);

    let same_a: usize = ea1.iter().zip(ea2.iter()).filter(|(a, b)| (*a - *b).abs() < 1e-5).count();
    let same_b: usize = eb1.iter().zip(eb2.iter()).filter(|(a, b)| (*a - *b).abs() < 1e-5).count();
    eprintln!("same(alpha 1st vs 2nd) {}/{}, same(beta 1st vs 2nd) {}/{}",
        same_a, ea1.len(), same_b, eb1.len());
}
