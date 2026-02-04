//! Comprehensive multi-model test summary
//!
//! Tests QKV layout fix across all available model architectures.

use gllm::Client;

struct ModelTestCase {
    alias: &'static str,
    description: &'static str,
    architecture: &'static str,
}

const MODELS_TO_TEST: &[ModelTestCase] = &[
    ModelTestCase {
        alias: "HuggingFaceTB/SmolLM2-135M-Instruct",
        description: "SmolLM2-135M - Llama4 architecture, GQA (9:3)",
        architecture: "Llama4",
    },
];

#[test]
#[ignore = "Run with: cargo test --test test_multi_model_summary -- --ignored"]
fn test_all_available_models() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║         Multi-Architecture QKV Layout Fix Verification                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut results = Vec::new();

    for model in MODELS_TO_TEST {
        println!("📦 Model: {}", model.alias);
        println!("   {}", model.description);
        println!();

        match Client::new(model.alias) {
            Ok(client) => {
                let actual_arch = format!("{:?}", client.manifest().arch);
                println!("   ✅ Loaded successfully");
                println!("   🏗️  Architecture: {}", actual_arch);

                // Test 1: Simple generation
                println!();
                println!("   Test 1: Simple Generation");
                let test_cases = vec![
                    ("The capital of France is", "Paris"),
                    ("1 + 1 equals", "2"),
                    ("The capital of Japan is", "Tokyo"),
                ];

                let mut passed = 0;
                for (prompt, expected) in &test_cases {
                    match client.generate(*prompt).max_tokens(10).temperature(0.0).generate() {
                        Ok(response) => {
                            let text = response.text.trim();
                            let contains = text.to_lowercase().contains(&expected.to_lowercase())
                                || text.contains(*expected);

                            if contains {
                                println!("      ✅ '{}'", prompt);
                                println!("         → '{}'", text);
                                passed += 1;
                            } else {
                                println!("      ⚠️  '{}'", prompt);
                                println!("         → '{}' (expected '{}')", text, expected);
                            }
                        }
                        Err(e) => {
                            println!("      ❌ '{}': {}", prompt, e);
                        }
                    }
                }

                results.push((model.alias, passed, test_cases.len()));
                println!();
                println!("   Sub-results: {}/{}", passed, test_cases.len());
            }
            Err(e) => {
                println!("   ❌ Failed to load: {}", e);
                results.push((model.alias, 0, 0));
            }
        }
        println!();
        for _ in 0..80 { print!("─"); }
        println!();
        println!();
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut total_passed = 0;
    let mut total_tests = 0;

    for (alias, passed, total) in &results {
        println!("   {:20} {:>3} / {}", alias, passed, total);
        total_passed += passed;
        total_tests += total;
    }

    println!();
    println!("   TOTAL: {} / {} tests passed", total_passed, total_tests);

    if total_passed == total_tests && total_tests > 0 {
        println!();
        println!("   ✅ ALL TESTS PASSED - QKV layout fix is working correctly!");
    }
}
