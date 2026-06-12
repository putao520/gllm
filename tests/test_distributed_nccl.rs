//! NCCL 分布式推理 E2E 测试
//!
//! 两台机器分布式推理 E2E 验证:
//! - 本地 (192.168.1.205, GTX 1060 6GB)
//! - 远程 (192.168.1.200, 5070 Ti)
//!
//! 运行方式:
//! ```bash
//! # 机器 0 (rank 0):
//! GLLM_NCCL_RANK=0 GLLM_NCCL_WORLD_SIZE=2 \
//!   LD_LIBRARY_PATH=~/.local/lib/python3.12/site-packages/nvidia/nccl/lib \
//!   cargo test --test test_distributed_nccl -- --test-threads=1
//!
//! # 机器 1 (rank 1):
//! GLLM_NCCL_RANK=1 GLLM_NCCL_WORLD_SIZE=2 \
//!   GLLM_NCCL_UNIQUE_ID=<from rank 0> \
//!   LD_LIBRARY_PATH=~/.local/lib/python3.12/site-packages/nvidia/nccl/lib \
//!   cargo test --test test_distributed_nccl -- --test-threads=1
//! ```
//!
//! 环境变量:
//! - GLLM_NCCL_RANK: 当前 rank (0 或 1)
//! - GLLM_NCCL_WORLD_SIZE: 总 rank 数 (2)
//! - GLLM_NCCL_UNIQUE_ID: rank 0 生成后广播给 rank 1 的 UniqueId (hex 编码)

#![cfg(feature = "nccl")]

use std::env;
use std::time::Duration;

use gllm_nccl::{
    get_unique_id, comm_init_rank, CommHandle, CollectiveOp, DType, ReduceOp, UniqueId,
};

fn get_rank() -> usize {
    env::var("GLLM_NCCL_RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn get_world_size() -> usize {
    env::var("GLLM_NCCL_WORLD_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

fn get_or_create_unique_id(rank: usize) -> UniqueId {
    if let Ok(hex) = env::var("GLLM_NCCL_UNIQUE_ID") {
        decode_unique_id(&hex)
    } else if rank == 0 {
        let id = get_unique_id().expect("NCCL get_unique_id failed");
        let hex = encode_unique_id(&id);
        eprintln!("[rank 0] UniqueId (share with rank 1):");
        eprintln!("  GLLM_NCCL_UNIQUE_ID={}", hex);
        id
    } else {
        panic!(
            "rank {} requires GLLM_NCCL_UNIQUE_ID from rank 0. \
             Set GLLM_NCCL_UNIQUE_ID env var.",
            rank
        );
    }
}

fn encode_unique_id(id: &UniqueId) -> String {
    id.bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn decode_unique_id(hex: &str) -> UniqueId {
    let bytes: Vec<u8> = (0..128)
        .map(|i| {
            u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .expect("invalid hex in GLLM_NCCL_UNIQUE_ID")
        })
        .collect();
    let mut arr = [0u8; 128];
    arr.copy_from_slice(&bytes);
    UniqueId { bytes: arr }
}

/// 测试 1: NCCL 初始化 — get_unique_id + comm_init_rank
///
/// 验证两台机器能成功建立通信组。
#[test]
fn test_nccl_init_rank() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    eprintln!("[rank {}] Initializing NCCL comm (world_size={})...", rank, world_size);
    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    assert_eq!(handle.rank(), rank);
    assert_eq!(handle.world_size(), world_size);
    eprintln!("[rank {}] NCCL comm initialized successfully.", rank);
}

/// 测试 2: CommHandle 属性验证
///
/// 验证 rank / world_size / topology / timeout 等基本属性。
#[test]
fn test_comm_handle_properties() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    // 基本属性
    assert_eq!(handle.rank(), rank);
    assert_eq!(handle.world_size(), world_size);

    // Topology
    let topo = handle.topology();
    assert!(!topo.devices.is_empty(), "topology should have devices");

    // Timeout 设置
    let custom_timeout = Duration::from_secs(300);
    handle.set_timeout(custom_timeout).expect("set_timeout failed");
    assert_eq!(handle.timeout(), custom_timeout);

    eprintln!("[rank {}] CommHandle properties verified: {} devices, timeout=300s",
              rank, topo.devices.len());
}

/// 测试 3: AllReduce F32 数值正确性
///
/// 每个 rank 提交 [1.0; N]，AllReduce Sum 后每个元素应为 world_size。
#[test]
fn test_all_reduce_f32_correctness() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    let count = 256;
    let mut sendbuf = vec![1.0f32; count];
    let mut recvbuf = vec![0.0f32; count];

    // 每个 rank 贡献 rank+1 的值，AllReduce Sum 后每个元素应为 sum(0..world_size)
    let contribution = (rank + 1) as f32;
    for elem in &mut sendbuf {
        *elem = contribution;
    }

    let expected = (0..world_size).map(|r| (r + 1) as f32).sum::<f32>();

    let future = handle
        .all_reduce::<f32>(&sendbuf, &mut recvbuf, ReduceOp::Sum)
        .expect("all_reduce failed");

    future.wait().expect("all_reduce wait failed");

    for (i, &val) in recvbuf.iter().enumerate() {
        assert!(
            (val - expected).abs() < 1e-3,
            "recvbuf[{}] = {}, expected {}",
            i, val, expected
        );
    }

    eprintln!("[rank {}] AllReduce F32 correctness verified: each element = {}",
              rank, expected);
}

/// 测试 4: AllReduce BF16 数值正确性
///
/// BF16 AllReduce 在 BF16 dtype 下执行，验证量化通信精度。
#[test]
fn test_all_reduce_bf16_correctness() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    let count = 128;
    let mut sendbuf = vec![half::bf16::from_f32(1.0); count];
    let mut recvbuf = vec![half::bf16::from_f32(0.0); count];

    let contribution = half::bf16::from_f32((rank + 1) as f32);
    for elem in &mut sendbuf {
        *elem = contribution;
    }

    let expected_f32 = (0..world_size).map(|r| (r + 1) as f32).sum::<f32>();
    let expected = half::bf16::from_f32(expected_f32);

    let future = handle
        .all_reduce::<half::bf16>(&sendbuf, &mut recvbuf, ReduceOp::Sum)
        .expect("all_reduce bf16 failed");

    future.wait().expect("all_reduce bf16 wait failed");

    for (i, &val) in recvbuf.iter().enumerate() {
        let diff = (val.to_f32() - expected.to_f32()).abs();
        assert!(
            diff < 0.1,
            "recvbuf[{}] = {:?} ({:.4}), expected {:.4}",
            i, val, val.to_f32(), expected.to_f32()
        );
    }

    eprintln!("[rank {}] AllReduce BF16 correctness verified.", rank);
}

/// 测试 5: AllGather F32 数值正确性
///
/// 每个 rank 贡献 [rank_val; N]，AllGather 后收集所有 rank 数据。
#[test]
fn test_all_gather_f32_correctness() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    let count_per_rank = 64;
    let mut sendbuf = vec![rank as f32; count_per_rank];
    let mut recvbuf = vec![0.0f32; count_per_rank * world_size];

    let future = handle
        .all_gather::<f32>(&sendbuf, &mut recvbuf)
        .expect("all_gather failed");

    future.wait().expect("all_gather wait failed");

    // 验证: recvbuf[r * count_per_rank .. (r+1) * count_per_rank] 应为 r.0
    for r in 0..world_size {
        let offset = r * count_per_rank;
        for i in 0..count_per_rank {
            let idx = offset + i;
            assert!(
                (recvbuf[idx] - r as f32).abs() < 1e-3,
                "recvbuf[{}] = {}, expected {}",
                idx, recvbuf[idx], r
            );
        }
    }

    eprintln!("[rank {}] AllGather F32 correctness verified.", rank);
}

/// 测试 6: ReduceScatter F32 数值正确性
///
/// 每个 rank 贡献 [world_size; total_count]，ReduceScatter Sum 后
/// 每个 rank 收到自己的段，值为 world_size * world_size。
#[test]
fn test_reduce_scatter_f32_correctness() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    let count_per_rank = 64;
    let total_count = count_per_rank * world_size;
    let mut sendbuf = vec![world_size as f32; total_count];
    let mut recvbuf = vec![0.0f32; count_per_rank];

    let future = handle
        .reduce_scatter::<f32>(&sendbuf, &mut recvbuf, ReduceOp::Sum)
        .expect("reduce_scatter failed");

    future.wait().expect("reduce_scatter wait failed");

    let expected = (world_size * world_size) as f32;
    for (i, &val) in recvbuf.iter().enumerate() {
        assert!(
            (val - expected).abs() < 1e-3,
            "recvbuf[{}] = {}, expected {}",
            i, val, expected
        );
    }

    eprintln!("[rank {}] ReduceScatter F32 correctness verified.", rank);
}

/// 测试 7: Broadcast F32 正确性
///
/// Root rank 广播 [42.0; N]，所有 rank 收到相同数据。
#[test]
fn test_broadcast_f32_correctness() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    let count = 128;
    let root = 0usize;
    let broadcast_val = if rank == root { 42.0f32 } else { 0.0f32 };
    let mut buf = vec![broadcast_val; count];

    let future = handle
        .broadcast::<f32>(&mut buf, root)
        .expect("broadcast failed");

    future.wait().expect("broadcast wait failed");

    for (i, &val) in buf.iter().enumerate() {
        assert!(
            (val - 42.0f32).abs() < 1e-3,
            "buf[{}] = {}, expected 42.0",
            i, val
        );
    }

    eprintln!("[rank {}] Broadcast F32 correctness verified.", rank);
}

/// 测试 8: CommHandle 毒化检测
///
/// 验证 CommHandle 的毒化检测机制在异步错误后正确标记。
#[test]
fn test_comm_handle_not_poisoned_after_init() {
    let rank = get_rank();
    let world_size = get_world_size();
    let unique_id = get_or_create_unique_id(rank);

    let handle = comm_init_rank(&unique_id, rank, world_size)
        .expect("comm_init_rank failed");

    // 新创建的 handle 不应该是毒化的
    handle.check_poisoned().expect("new handle should not be poisoned");
    assert!(!handle.is_gpu_opt_active(), "GPU opt should not be active by default");

    eprintln!("[rank {}] CommHandle poison check passed.", rank);
}
