/// SIGSEGV 压力测试 - 检测 wgpu 清理的稳定性
///
/// 用法:
///   cargo run --release --bin stress_test -- --case 1 --iterations 10

use gllm::Client;
use std::env;
use std::process;

struct TestStats {
    case: usize,
    iterations: usize,
    successes: usize,
    failures: usize,
    sigsegv_count: usize,
}

impl TestStats {
    fn new(case: usize, iterations: usize) -> Self {
        Self {
            case,
            iterations,
            successes: 0,
            failures: 0,
            sigsegv_count: 0,
        }
    }

    fn success_rate(&self) -> f32 {
        if self.iterations == 0 {
            0.0
        } else {
            (self.successes as f32) / (self.iterations as f32) * 100.0
        }
    }

    fn print_report(&self) {
        let sep = "=".repeat(60);
        println!("\n{}", sep);
        println!("📊 SIGSEGV 压力测试报告");
        println!("{}", sep);
        println!("测试用例: {}", self.case);
        println!("总次数: {}", self.iterations);
        println!("成功: {} ✅", self.successes);
        println!("失败: {} ❌", self.failures);
        println!("SIGSEGV: {} 💥", self.sigsegv_count);
        println!("成功率: {:.1}%", self.success_rate());
        println!("{}\n", sep);
    }
}

/// 用例 1: 频繁创建销毁 Client
fn test_case_1_rapid_create_destroy(iterations: usize) -> TestStats {
    let mut stats = TestStats::new(1, iterations);
    println!("🧪 用例 1: 频繁创建销毁 Client ({}次)", iterations);

    for i in 1..=iterations {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _client = Client::new("bge-small-en")
                .expect("Failed to create client");
        })) {
            Ok(_) => {
                stats.successes += 1;
                print!(".");
            }
            Err(_) => {
                stats.failures += 1;
                stats.sigsegv_count += 1;
                print!("X");
            }
        }
        if i % 50 == 0 {
            println!(" {}/{}", i, iterations);
        }
    }
    println!();
    stats
}

/// 用例 2: 多线程并发使用
fn test_case_2_concurrent_threads(iterations: usize) -> TestStats {
    let mut stats = TestStats::new(2, iterations);
    println!("🧪 用例 2: 多线程并发使用 ({}次)", iterations);

    for i in 1..=iterations {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut handles = vec![];

            for _ in 0..4 {
                let handle = std::thread::spawn(|| {
                    let _client = Client::new("bge-small-en")
                        .expect("Failed to create client");
                    std::thread::sleep(std::time::Duration::from_millis(10));
                });
                handles.push(handle);
            }

            for handle in handles {
                let _ = handle.join();
            }
        })) {
            Ok(_) => {
                stats.successes += 1;
                print!(".");
            }
            Err(_) => {
                stats.failures += 1;
                stats.sigsegv_count += 1;
                print!("X");
            }
        }
        if i % 20 == 0 {
            println!(" {}/{}", i, iterations);
        }
    }
    println!();
    stats
}

/// 用例 3: 大量推理后快速退出
fn test_case_3_heavy_inference_quick_exit(iterations: usize) -> TestStats {
    let mut stats = TestStats::new(3, iterations);
    println!("🧪 用例 3: 大量推理后快速退出 ({}次)", iterations);

    for i in 1..=iterations {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let client = Client::new("bge-small-en")
                .expect("Failed to create client");

            let texts = vec![
                "The quick brown fox jumps over the lazy dog".to_string();
                100
            ];
            let response = client
                .embeddings(texts)
                .generate()
                .expect("Failed to compute embeddings");

            assert!(!response.embeddings.is_empty());
        })) {
            Ok(_) => {
                stats.successes += 1;
                print!(".");
            }
            Err(_) => {
                stats.failures += 1;
                stats.sigsegv_count += 1;
                print!("X");
            }
        }
        if i % 10 == 0 {
            println!(" {}/{}", i, iterations);
        }
    }
    println!();
    stats
}

/// 用例 4: 不同 Backend 切换
fn test_case_4_backend_switching(_iterations: usize) -> TestStats {
    let mut stats = TestStats::new(4, 1);
    println!("🧪 用例 4: 不同 Backend 切换");

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for _ in 0..3 {
            let _client = Client::new("bge-small-en")
                .expect("Failed to create client");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    })) {
        Ok(_) => {
            stats.successes = 1;
            println!("✅ Backend switching test passed");
        }
        Err(_) => {
            stats.failures = 1;
            stats.sigsegv_count = 1;
            println!("❌ Backend switching test SIGSEGV");
        }
    }
    stats
}

/// 用例 5: Panic 清理
fn test_case_5_panic_cleanup(iterations: usize) -> TestStats {
    let mut stats = TestStats::new(5, iterations);
    println!("🧪 用例 5: Panic 清理 ({}次)", iterations);

    for i in 1..=iterations {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _client = Client::new("bge-small-en")
                .expect("Failed to create client");

            if i % 5 == 0 {
                panic!("Intentional panic for cleanup testing");
            }
        })) {
            Ok(_) => {
                stats.successes += 1;
                print!(".");
            }
            Err(_) => {
                stats.successes += 1;
                print!("P");
            }
        }
        if i % 20 == 0 {
            println!(" {}/{}", i, iterations);
        }
    }
    println!();
    stats
}

/// 用例 6: 长时间运行
fn test_case_6_long_running(iterations: usize) -> TestStats {
    let mut stats = TestStats::new(6, iterations);
    println!("🧪 用例 6: 长时间运行 ({}次推理)", iterations);

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = Client::new("bge-small-en")
            .expect("Failed to create client");

        for i in 1..=iterations {
            let response = client
                .embeddings(vec!["Test text"])
                .generate()
                .expect("Failed to compute embedding");

            assert!(!response.embeddings.is_empty());

            if i % 100 == 0 {
                print!(".");
                if i % 500 == 0 {
                    println!(" {}/{}", i, iterations);
                }
            }
        }
    })) {
        Ok(_) => {
            stats.successes = 1;
            println!();
            println!("✅ Long-running test completed");
        }
        Err(_) => {
            stats.failures = 1;
            stats.sigsegv_count = 1;
            println!();
            println!("❌ Long-running test SIGSEGV");
        }
    }
    stats
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut case = None;
    let mut iterations = 10;

    for i in 0..args.len() {
        match args[i].as_str() {
            "--case" => {
                if i + 1 < args.len() {
                    case = args[i + 1].parse().ok();
                }
            }
            "--iterations" => {
                if i + 1 < args.len() {
                    iterations = args[i + 1].parse().unwrap_or(10);
                }
            }
            _ => {}
        }
    }

    if case.is_none() {
        eprintln!("用法: stress_test --case <1-6> --iterations <N>");
        eprintln!("用例:");
        eprintln!("  1: 频繁创建销毁");
        eprintln!("  2: 多线程并发");
        eprintln!("  3: 大量推理后快速退出");
        eprintln!("  4: Backend 切换");
        eprintln!("  5: Panic 清理");
        eprintln!("  6: 长时间运行");
        process::exit(1);
    }

    let case = case.unwrap();
    println!("开始 SIGSEGV 压力测试\n");
    println!("参数: case={}, iterations={}\n", case, iterations);

    let stats = match case {
        1 => test_case_1_rapid_create_destroy(iterations),
        2 => test_case_2_concurrent_threads(iterations),
        3 => test_case_3_heavy_inference_quick_exit(iterations),
        4 => test_case_4_backend_switching(iterations),
        5 => test_case_5_panic_cleanup(iterations),
        6 => test_case_6_long_running(iterations),
        _ => {
            eprintln!("❌ 未知的测试用例: {}", case);
            process::exit(1);
        }
    };

    stats.print_report();

    if stats.sigsegv_count > 0 {
        eprintln!("⚠️  检测到 {} 次 SIGSEGV!", stats.sigsegv_count);
        process::exit(1);
    }

    println!("✅ 所有测试通过!");
    process::exit(0);
}
