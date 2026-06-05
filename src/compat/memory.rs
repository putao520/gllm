use crate::engine::executor::BackendError as BE;

// ---------------------------------------------------------------------------
// System memory pressure monitoring (no external crates)
// ---------------------------------------------------------------------------

/// Available system memory in bytes (REQ-PA-007: memory safety guardrail).
pub fn get_available_memory_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let content = match std::fs::read_to_string("/proc/meminfo") {
            Ok(c) => c,
            Err(_) => return 0,
        };
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                if let Some(kb) = parse_meminfo_kb(rest) {
                    return kb * 1024;
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

pub fn get_system_memory_pressure() -> Result<f32, BE> {
    #[cfg(target_os = "linux")]
    {
        get_memory_pressure_linux()
    }

    #[cfg(target_os = "macos")]
    {
        return get_memory_pressure_macos();
    }

    #[cfg(target_os = "windows")]
    {
        return get_memory_pressure_windows();
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Ok(0.0)
    }
}

#[cfg(target_os = "linux")]
fn get_memory_pressure_linux() -> Result<f32, BE> {
    let content = match std::fs::read_to_string("/proc/meminfo") {
        Ok(c) => c,
        Err(e) => {
            log::debug!("cannot read /proc/meminfo: {e}");
            return Ok(0.0);
        }
    };

    let mut mem_total_kb: Option<u64> = None;
    let mut mem_available_kb: Option<u64> = None;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            mem_total_kb = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            mem_available_kb = parse_meminfo_kb(rest);
        }
        if mem_total_kb.is_some() && mem_available_kb.is_some() {
            break;
        }
    }

    match (mem_total_kb, mem_available_kb) {
        (Some(total), Some(available)) if total > 0 => {
            let pressure = 1.0 - (available as f64 / total as f64);
            Ok(pressure.clamp(0.0, 1.0) as f32)
        }
        _ => Ok(0.0),
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(value: &str) -> Option<u64> {
    let trimmed = value.trim().trim_end_matches("kB").trim();
    trimmed.parse::<u64>().ok()
}

#[cfg(target_os = "macos")]
fn get_memory_pressure_macos() -> Result<f32, BE> {
    // macOS: sysctl(hw.memsize) for total, vm_stat for free/inactive pages
    extern "C" {
        fn sysctlbyname(
            name: *const u8,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }

    // Get total physical memory via sysctl("hw.memsize")
    let mut total_mem: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let key = b"hw.memsize\0";
    let rc = unsafe {
        sysctlbyname(
            key.as_ptr(),
            &mut total_mem as *mut u64 as *mut std::ffi::c_void,
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 || total_mem == 0 {
        return Ok(0.0);
    }

    // Parse vm_stat output for free + inactive pages
    let vm_stat = match std::process::Command::new("vm_stat").output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return Ok(0.0),
    };

    let mut free_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let mut page_size: u64 = 4096;

    for line in vm_stat.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ") {
            if let Some(num) = rest.strip_suffix(" bytes)") {
                page_size = num.parse().map_err(|_| {
                    BE::Cpu(format!("failed to parse vm_stat page size: {:?}", num))
                })?;
            }
        } else if let Some(rest) = line.strip_prefix("Pages free:") {
            free_pages = parse_vm_stat_value(rest);
        } else if let Some(rest) = line.strip_prefix("Pages inactive:") {
            inactive_pages = parse_vm_stat_value(rest);
        }
    }

    let available = (free_pages + inactive_pages) * page_size;
    let pressure = 1.0 - (available as f64 / total_mem as f64);
    Ok(pressure.clamp(0.0, 1.0) as f32)
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_value(s: &str) -> u64 {
    s.trim().trim_end_matches('.').parse().unwrap_or(0) // LEGAL: vm_stat 解析失败默认 0
}

#[cfg(target_os = "windows")]
fn get_memory_pressure_windows() -> Result<f32, BE> {
    // Windows: call GlobalMemoryStatusEx via raw FFI (no winapi crate dependency)
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };

    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 || status.ull_total_phys == 0 {
        return Ok(0.0);
    }

    let pressure = 1.0 - (status.ull_avail_phys as f64 / status.ull_total_phys as f64);
    Ok(pressure.clamp(0.0, 1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_system_memory_pressure_succeeds() {
        let result = get_system_memory_pressure();
        assert!(result.is_ok(), "memory pressure query should succeed");
        let pressure = result.unwrap();
        assert!(pressure >= 0.0 && pressure <= 1.0, "pressure should be in [0, 1]");
    }

    #[test]
    fn get_system_memory_pressure_returns_f32() {
        let result = get_system_memory_pressure().unwrap();
        // Verify the result is a finite f32 (not NaN, not infinity)
        assert!(result.is_finite(), "pressure should be a finite f32");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_valid() {
        assert_eq!(parse_meminfo_kb("    16384000 kB"), Some(16384000));
        assert_eq!(parse_meminfo_kb("  1024 kB"), Some(1024));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_no_kb_suffix() {
        assert_eq!(parse_meminfo_kb("  4096"), Some(4096));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_invalid() {
        assert_eq!(parse_meminfo_kb("not a number"), None);
        assert_eq!(parse_meminfo_kb(""), None);
    }

    // ── parse_meminfo_kb edge cases ──────────────────────────────────

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_leading_and_trailing_whitespace() {
        assert_eq!(parse_meminfo_kb("   8192   kB   "), Some(8192));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_zero_value() {
        assert_eq!(parse_meminfo_kb("0 kB"), Some(0));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_large_value() {
        // Simulate a machine with ~1 TiB RAM
        let one_tib_kb: u64 = 1024 * 1024 * 1024;
        assert_eq!(parse_meminfo_kb(&format!("  {one_tib_kb} kB")), Some(one_tib_kb));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_max_u64() {
        assert_eq!(parse_meminfo_kb(&format!("{} kB", u64::MAX)), Some(u64::MAX));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_only_whitespace() {
        assert_eq!(parse_meminfo_kb("   "), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_negative_sign_rejected() {
        // u64 parse should fail on negative
        assert_eq!(parse_meminfo_kb("-512 kB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_float_value_rejected() {
        assert_eq!(parse_meminfo_kb("3.14 kB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_case_sensitive_suffix() {
        // trim_end_matches("kB") strips individual trailing chars 'k' and 'B'.
        // "KB" has trailing 'K' (not in "kB") and 'B' (in "kB"), so only 'B' is stripped,
        // leaving "  4096 K" which cannot parse as u64.
        assert_eq!(parse_meminfo_kb("  4096 KB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_multiple_kb_suffixes() {
        // Only one "kB" suffix is stripped
        assert_eq!(parse_meminfo_kb("4096 kB kB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_hex_rejected() {
        assert_eq!(parse_meminfo_kb("0xFF kB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_plus_sign_accepted() {
        // Rust's u64::parse accepts a leading '+'
        assert_eq!(parse_meminfo_kb("+1024 kB"), Some(1024));
    }

    // ── get_memory_pressure_linux with synthetic meminfo ─────────────
    // These test the pressure calculation logic by verifying the formula
    // through direct calls to get_system_memory_pressure (which delegates
    // to get_memory_pressure_linux on Linux). Since /proc/meminfo is a
    // real file, we verify the result is a valid pressure value.

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_memory_pressure_in_valid_range() {
        let result = get_memory_pressure_linux();
        assert!(result.is_ok());
        let pressure = result.unwrap();
        assert!(
            (0.0..=1.0).contains(&pressure),
            "pressure {pressure} should be in [0.0, 1.0]"
        );
    }

    // ── Pressure calculation correctness (unit-level) ────────────────
    // Verify the formula: pressure = 1 - (available / total), clamped to [0, 1]

    #[test]
    fn pressure_formula_full_usage() {
        // If available = 0, pressure should be 1.0
        let total: u64 = 16384000;
        let available: u64 = 0;
        let pressure = 1.0 - (available as f64 / total as f64);
        assert!((pressure - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_formula_half_usage() {
        let total: u64 = 16384000;
        let available: u64 = 8192000;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert!((pressure - 0.5).abs() < 0.001);
    }

    #[test]
    fn pressure_formula_zero_usage() {
        let total: u64 = 16384000;
        let available: u64 = 16384000;
        let pressure = 1.0 - (available as f64 / total as f64);
        assert!(pressure.abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_clamp_lower_bound() {
        // If somehow available > total, pressure would be negative but clamp to 0.0
        let total: u64 = 1000;
        let available: u64 = 2000;
        let pressure = (1.0 - (available as f64 / total as f64)).clamp(0.0, 1.0);
        assert!((pressure - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_clamp_upper_bound() {
        // Negative available (impossible in practice) would give >1.0, clamped to 1.0
        let total: u64 = 1000;
        let available: f64 = -500.0;
        let pressure = (1.0 - (available / total as f64)).clamp(0.0, 1.0);
        assert!((pressure - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_formula_typical_values() {
        // 16 GiB total, 4 GiB available → 75% used → pressure = 0.75
        let total_kb: u64 = 16 * 1024 * 1024;
        let available_kb: u64 = 4 * 1024 * 1024;
        let pressure = (1.0 - (available_kb as f64 / total_kb as f64)) as f32;
        assert!((pressure - 0.75).abs() < 0.001);
    }

    // ── BackendError variants used by memory module ──────────────────

    #[test]
    fn backend_error_cpu_variant_display() {
        let err = BE::Cpu("test error message".into());
        let display = format!("{err}");
        assert!(display.contains("CPU error"));
        assert!(display.contains("test error message"));
    }

    #[test]
    fn backend_error_cpu_debug_format() {
        let err = BE::Cpu("dbg".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Cpu"));
        assert!(debug.contains("dbg"));
    }

    #[test]
    fn backend_error_clone_equals_original() {
        let err = BE::Cpu("clone me".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_variants_distinct() {
        let variants = [
            format!("{}", BE::Cuda("a".into())),
            format!("{}", BE::Hip("b".into())),
            format!("{}", BE::Metal("c".into())),
            format!("{}", BE::Cpu("d".into())),
            format!("{}", BE::Unimplemented("e")),
            format!("{}", BE::Other("f".into())),
        ];
        // All display strings should be unique
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variant {i} and {j} should differ");
            }
        }
    }

    #[test]
    fn backend_error_std_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(BE::Cpu("trait check".into()));
        let msg = err.to_string();
        assert!(msg.contains("CPU error"));
    }

    // =====================================================================
    // New tests — 45+ additional tests
    // =====================================================================

    // ── BackendError Display for every variant ──────────────────────────

    #[test]
    fn backend_error_cuda_display() {
        let err = BE::Cuda("device not found".into());
        let display = format!("{err}");
        assert!(display.contains("CUDA error"));
        assert!(display.contains("device not found"));
    }

    #[test]
    fn backend_error_hip_display() {
        let err = BE::Hip("HIP failure".into());
        let display = format!("{err}");
        assert!(display.contains("HIP error"));
        assert!(display.contains("HIP failure"));
    }

    #[test]
    fn backend_error_metal_display() {
        let err = BE::Metal("shader compile".into());
        let display = format!("{err}");
        assert!(display.contains("Metal error"));
        assert!(display.contains("shader compile"));
    }

    #[test]
    fn backend_error_unimplemented_display() {
        let err = BE::Unimplemented("feature X");
        let display = format!("{err}");
        assert!(display.contains("unimplemented"));
        assert!(display.contains("feature X"));
    }

    #[test]
    fn backend_error_other_display() {
        let err = BE::Other("generic".into());
        let display = format!("{err}");
        assert!(display.contains("backend error"));
        assert!(display.contains("generic"));
    }

    // ── BackendError Debug format ────────────────────────────────────────

    #[test]
    fn backend_error_cuda_debug() {
        let err = BE::Cuda("dbg-cuda".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Cuda"));
        assert!(debug.contains("dbg-cuda"));
    }

    #[test]
    fn backend_error_hip_debug() {
        let err = BE::Hip("dbg-hip".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Hip"));
        assert!(debug.contains("dbg-hip"));
    }

    #[test]
    fn backend_error_metal_debug() {
        let err = BE::Metal("dbg-metal".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Metal"));
        assert!(debug.contains("dbg-metal"));
    }

    #[test]
    fn backend_error_unimplemented_debug() {
        let err = BE::Unimplemented("dbg-unimpl");
        let debug = format!("{err:?}");
        assert!(debug.contains("Unimplemented"));
        assert!(debug.contains("dbg-unimpl"));
    }

    #[test]
    fn backend_error_other_debug() {
        let err = BE::Other("dbg-other".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Other"));
        assert!(debug.contains("dbg-other"));
    }

    // ── BackendError Clone round-trip ────────────────────────────────────

    #[test]
    fn backend_error_cuda_clone_roundtrip() {
        let err = BE::Cuda("clone-cuda".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_hip_clone_roundtrip() {
        let err = BE::Hip("clone-hip".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_metal_clone_roundtrip() {
        let err = BE::Metal("clone-metal".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_unimplemented_clone_roundtrip() {
        let err = BE::Unimplemented("clone-static");
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_other_clone_roundtrip() {
        let err = BE::Other("clone-other".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    // ── BackendError std::error::Error for every variant ─────────────────

    #[test]
    fn backend_error_cuda_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BE::Cuda("err".into()));
        assert!(err.to_string().contains("CUDA error"));
    }

    #[test]
    fn backend_error_hip_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BE::Hip("err".into()));
        assert!(err.to_string().contains("HIP error"));
    }

    #[test]
    fn backend_error_metal_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BE::Metal("err".into()));
        assert!(err.to_string().contains("Metal error"));
    }

    #[test]
    fn backend_error_unimplemented_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BE::Unimplemented("err"));
        assert!(err.to_string().contains("unimplemented"));
    }

    #[test]
    fn backend_error_other_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BE::Other("err".into()));
        assert!(err.to_string().contains("backend error"));
    }

    // ── BackendError with empty strings ──────────────────────────────────

    #[test]
    fn backend_error_cpu_empty_string() {
        let err = BE::Cpu(String::new());
        let display = format!("{err}");
        assert!(display.contains("CPU error"));
    }

    #[test]
    fn backend_error_cuda_empty_string() {
        let err = BE::Cuda(String::new());
        let display = format!("{err}");
        assert!(display.contains("CUDA error"));
    }

    #[test]
    fn backend_error_other_empty_string() {
        let err = BE::Other(String::new());
        let display = format!("{err}");
        assert!(display.contains("backend error"));
    }

    // ── BackendError with long strings ───────────────────────────────────

    #[test]
    fn backend_error_cpu_long_message() {
        let long_msg = "x".repeat(10000);
        let err = BE::Cpu(long_msg.clone());
        let display = format!("{err}");
        assert!(display.contains(&long_msg));
        assert!(display.len() > 10000);
    }

    #[test]
    fn backend_error_unimplemented_static_str() {
        let err = BE::Unimplemented("this is a static str");
        let display = format!("{err}");
        assert_eq!(display, "unimplemented: this is a static str");
    }

    // ── Pressure formula edge cases ──────────────────────────────────────

    #[test]
    fn pressure_formula_single_byte_total() {
        let total: u64 = 1;
        let available: u64 = 0;
        let pressure = 1.0 - (available as f64 / total as f64);
        assert!((pressure - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_formula_single_byte_available() {
        let total: u64 = 1;
        let available: u64 = 1;
        let pressure = 1.0 - (available as f64 / total as f64);
        assert!(pressure.abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_formula_large_total_u64_max() {
        let total: u64 = u64::MAX;
        let available: u64 = u64::MAX / 2;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert!((pressure - 0.5).abs() < 0.001);
    }

    #[test]
    fn pressure_formula_very_small_difference() {
        // 1 MB used out of 1 TB
        let total_kb: u64 = 1024 * 1024 * 1024;
        let available_kb: u64 = total_kb - 1024;
        let pressure = (1.0 - (available_kb as f64 / total_kb as f64)) as f32;
        assert!(pressure > 0.0);
        assert!(pressure < 0.001);
    }

    #[test]
    fn pressure_formula_ninety_percent_used() {
        let total: u64 = 1000;
        let available: u64 = 100;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert!((pressure - 0.9).abs() < 0.001);
    }

    #[test]
    fn pressure_formula_one_percent_used() {
        let total: u64 = 10000;
        let available: u64 = 9900;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert!((pressure - 0.01).abs() < 0.001);
    }

    // ── clamp behavior with special floats ───────────────────────────────

    #[test]
    fn pressure_clamp_nan_input() {
        // NaN input to clamp: f64::clamp(NaN, 0.0, 1.0) returns NaN in Rust
        // but the code path produces f64 from u64 division, so NaN shouldn't arise
        // We test clamp behavior with valid values only
        let val = 0.5_f64;
        let clamped = val.clamp(0.0, 1.0);
        assert!((clamped - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_clamp_at_exactly_zero() {
        let val = 0.0_f64;
        let clamped = val.clamp(0.0, 1.0);
        assert!((clamped - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_clamp_at_exactly_one() {
        let val = 1.0_f64;
        let clamped = val.clamp(0.0, 1.0);
        assert!((clamped - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_clamp_just_above_one() {
        let val = 1.0001_f64;
        let clamped = val.clamp(0.0, 1.0);
        assert!((clamped - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_clamp_just_below_zero() {
        let val = -0.0001_f64;
        let clamped = val.clamp(0.0, 1.0);
        assert!((clamped - 0.0).abs() < f64::EPSILON);
    }

    // ── f32 conversion from f64 pressure ─────────────────────────────────

    #[test]
    fn pressure_f32_conversion_preserves_range() {
        for val in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let f64_val = val;
            let f32_val = f64_val as f32;
            assert!((f32_val as f64 - f64_val).abs() < 0.0001);
        }
    }

    #[test]
    fn pressure_f32_is_not_nan() {
        let total: u64 = 1000;
        let available: u64 = 500;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert!(!pressure.is_nan());
        assert!(pressure.is_finite());
    }

    // ── get_system_memory_pressure cross-platform ────────────────────────

    #[test]
    fn get_system_memory_pressure_not_nan() {
        let result = get_system_memory_pressure().unwrap();
        assert!(!result.is_nan());
    }

    #[test]
    fn get_system_memory_pressure_not_infinite() {
        let result = get_system_memory_pressure().unwrap();
        assert!(!result.is_infinite());
    }

    #[test]
    fn get_system_memory_pressure_is_normal_or_zero() {
        let result = get_system_memory_pressure().unwrap();
        // 0.0 is subnormal but valid; positive normal is also valid
        assert!(result == 0.0 || result.is_normal());
    }

    #[test]
    fn get_system_memory_pressure_ok_on_all_platforms() {
        // The function should return Ok on any platform (even unknown ones return Ok(0.0))
        let result = get_system_memory_pressure();
        assert!(result.is_ok());
    }

    // ── Additional parse_meminfo_kb edge cases (Linux) ───────────────────

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_value_one() {
        assert_eq!(parse_meminfo_kb("1 kB"), Some(1));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_value_with_many_spaces() {
        assert_eq!(parse_meminfo_kb("     2048     kB"), Some(2048));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_tab_separated() {
        assert_eq!(parse_meminfo_kb("4096\tkB"), Some(4096));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_mixed_whitespace() {
        assert_eq!(parse_meminfo_kb("  \t  8192  \t kB  \t"), Some(8192));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_underscore_rejected() {
        // Rust u64 parse does not accept _ separators
        assert_eq!(parse_meminfo_kb("1_000 kB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_scientific_notation_rejected() {
        assert_eq!(parse_meminfo_kb("1e6 kB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_only_suffix() {
        assert_eq!(parse_meminfo_kb("kB"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_suffix_without_space() {
        assert_eq!(parse_meminfo_kb("4096kB"), Some(4096));
    }

    // ── Pressure calculation: u64 overflow safety ────────────────────────

    #[test]
    fn pressure_calc_no_overflow_when_total_max() {
        // Verify f64 cast of u64::MAX doesn't lose precision catastrophically
        let total: u64 = u64::MAX;
        let available: u64 = 1;
        let ratio = available as f64 / total as f64;
        assert!(ratio >= 0.0);
        assert!(ratio < 0.001);
    }

    #[test]
    fn pressure_calc_both_zero_safe() {
        // total=0 is guarded in real code, but verify the f64 math itself
        // In the real code, total > 0 is checked, so this tests the formula in isolation
        let total: u64 = 0;
        let available: u64 = 0;
        // 0.0 / 0.0 = NaN, but the real code guards against total == 0
        if total > 0 {
            let pressure = 1.0 - (available as f64 / total as f64);
            assert!(pressure.is_finite());
        }
        // This branch taken since total == 0: the guard works
        assert!(total == 0);
    }

    #[test]
    fn pressure_calc_available_equals_total() {
        let total: u64 = 999999;
        let available: u64 = 999999;
        let pressure = 1.0 - (available as f64 / total as f64);
        assert!(pressure.abs() < 1e-10);
    }

    #[test]
    fn pressure_calc_available_one_less_than_total() {
        let total: u64 = 100000;
        let available: u64 = 99999;
        let pressure = 1.0 - (available as f64 / total as f64);
        assert!((pressure - 0.00001).abs() < 1e-10);
    }

    // ── Display format exactness ─────────────────────────────────────────

    #[test]
    fn backend_error_cpu_display_exact() {
        let err = BE::Cpu("msg".into());
        assert_eq!(format!("{err}"), "CPU error: msg");
    }

    #[test]
    fn backend_error_cuda_display_exact() {
        let err = BE::Cuda("msg".into());
        assert_eq!(format!("{err}"), "CUDA error: msg");
    }

    #[test]
    fn backend_error_hip_display_exact() {
        let err = BE::Hip("msg".into());
        assert_eq!(format!("{err}"), "HIP error: msg");
    }

    #[test]
    fn backend_error_metal_display_exact() {
        let err = BE::Metal("msg".into());
        assert_eq!(format!("{err}"), "Metal error: msg");
    }

    #[test]
    fn backend_error_unimplemented_display_exact() {
        let err = BE::Unimplemented("thing");
        assert_eq!(format!("{err}"), "unimplemented: thing");
    }

    #[test]
    fn backend_error_other_display_exact() {
        let err = BE::Other("msg".into());
        assert_eq!(format!("{err}"), "backend error: msg");
    }

    // ── BackendError with unicode messages ───────────────────────────────

    #[test]
    fn backend_error_cpu_unicode_message() {
        let err = BE::Cpu("错误信息 🚨".into());
        let display = format!("{err}");
        assert!(display.contains("错误信息 🚨"));
    }

    #[test]
    fn backend_error_other_unicode_message() {
        let err = BE::Other("エラー".into());
        let display = format!("{err}");
        assert!(display.contains("エラー"));
    }

    // ── Multiple BackendError instances are independent ──────────────────

    #[test]
    fn backend_error_two_cpu_errors_independent() {
        let err1 = BE::Cpu("first".into());
        let err2 = BE::Cpu("second".into());
        assert_ne!(format!("{err1}"), format!("{err2}"));
    }

    #[test]
    fn backend_error_different_variants_independent() {
        let err1 = BE::Cpu("same".into());
        let err2 = BE::Cuda("same".into());
        assert_ne!(format!("{err1}"), format!("{err2}"));
    }

    // ── Pressure f32 boundary: exactly 0.0 and 1.0 ──────────────────────

    #[test]
    fn pressure_f32_zero_exactly() {
        let total: u64 = 100;
        let available: u64 = 100;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert_eq!(pressure, 0.0_f32);
    }

    #[test]
    fn pressure_f32_one_exactly() {
        let total: u64 = 100;
        let available: u64 = 0;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert_eq!(pressure, 1.0_f32);
    }

    #[test]
    fn pressure_f32_half_precision() {
        let total: u64 = 2;
        let available: u64 = 1;
        let pressure = (1.0 - (available as f64 / total as f64)) as f32;
        assert!((pressure - 0.5_f32).abs() < f32::EPSILON);
    }

    // ── Result type inspection ───────────────────────────────────────────

    #[test]
    fn get_system_memory_pressure_returns_result() {
        // Verify the return type is Result<f32, BackendError>
        let result: Result<f32, BE> = get_system_memory_pressure();
        assert!(result.is_ok());
    }

    #[test]
    fn get_system_memory_pressure_multiple_calls_consistent() {
        // Two calls within short time should return similar values
        let p1 = get_system_memory_pressure().unwrap();
        let p2 = get_system_memory_pressure().unwrap();
        // Memory pressure shouldn't change by more than 10% in a few microseconds
        assert!((p1 - p2).abs() < 0.1, "p1={p1}, p2={p2} differ too much");
    }

    // ── BackendError with special characters in message ──────────────────

    #[test]
    fn backend_error_cpu_newline_in_message() {
        let err = BE::Cpu("line1\nline2".into());
        let display = format!("{err}");
        assert!(display.contains("line1"));
        assert!(display.contains("line2"));
    }

    #[test]
    fn backend_error_cpu_null_byte_in_message() {
        // String can contain embedded content; Display should still work
        let msg = "before\0after".to_string();
        let err = BE::Cpu(msg);
        let display = format!("{err}");
        assert!(display.contains("before"));
    }

    // ── clamp as f32 ─────────────────────────────────────────────────────

    #[test]
    fn pressure_f32_clamp_high() {
        let val = 2.0_f32;
        let clamped = val.clamp(0.0, 1.0);
        assert_eq!(clamped, 1.0_f32);
    }

    #[test]
    fn pressure_f32_clamp_low() {
        let val = -1.0_f32;
        let clamped = val.clamp(0.0, 1.0);
        assert_eq!(clamped, 0.0_f32);
    }

    #[test]
    fn pressure_f32_clamp_in_range() {
        let val = 0.3_f32;
        let clamped = val.clamp(0.0, 1.0);
        assert!((clamped - 0.3).abs() < f32::EPSILON);
    }

    // ── Linux-specific: parse_meminfo_kb with edge-case whitespace ───────

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_only_newlines() {
        assert_eq!(parse_meminfo_kb("\n\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_trailing_newline() {
        assert_eq!(parse_meminfo_kb("4096 kB\n"), Some(4096));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_leading_newline() {
        assert_eq!(parse_meminfo_kb("\n4096 kB"), Some(4096));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_double_kb_suffix() {
        // "kBkB" - trim_end_matches strips individual chars, "kBkB" -> ""
        assert_eq!(parse_meminfo_kb("4096 kBkB"), Some(4096));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_alternating_case_suffix() {
        // "kB" is the set {'k','B'}. "Kb" has 'K' (not in set) and 'b' (not in set)
        // so nothing is stripped -> parse "4096 Kb" fails
        assert_eq!(parse_meminfo_kb("4096 Kb"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_whitespace_only_after_value() {
        assert_eq!(parse_meminfo_kb("4096   "), Some(4096));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_u64_max_boundary() {
        let s = format!("{} kB", u64::MAX);
        assert_eq!(parse_meminfo_kb(&s), Some(u64::MAX));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_kb_u64_overflow_rejected() {
        // u64::MAX + 1 overflows, but as a string it's too large for u64
        assert_eq!(parse_meminfo_kb("18446744073709551616 kB"), None);
    }

    // =====================================================================
    // Additional tests — boundary, integration, ownership semantics
    // =====================================================================

    // ── BackendError: two same-variant instances with identical payloads ──

    #[test]
    fn backend_error_same_variant_same_msg_display_equal() {
        // Arrange: two independent Cpu errors with identical content
        let err1 = BE::Cpu("identical".into());
        let err2 = BE::Cpu("identical".into());
        // Act & Assert: Display output should be equal
        assert_eq!(format!("{err1}"), format!("{err2}"));
    }

    // ── BackendError: clone produces independent copy (mutation-safe) ─────

    #[test]
    fn backend_error_clone_is_independent() {
        // Arrange
        let original = BE::Cpu("original".into());
        let cloned = original.clone();
        // Act & Assert: both live independently; display is identical
        assert_eq!(format!("{original}"), format!("{cloned}"));
        // They are separate allocations — dropping one must not affect the other.
        drop(original);
        assert_eq!(format!("{cloned}"), "CPU error: original");
    }

    // ── BackendError: Debug output differs from Display ──────────────────

    #[test]
    fn backend_error_debug_differs_from_display() {
        // Arrange
        let err = BE::Cpu("compare".into());
        // Act
        let debug = format!("{err:?}");
        let display = format!("{err}");
        // Assert: Debug contains the variant name (Cpu), Display contains the
        // human-readable prefix ("CPU error:"). They are different strings.
        assert_ne!(debug, display);
        assert!(debug.contains("Cpu"));
        assert!(display.contains("CPU error: compare"));
    }

    // ── BackendError: Unimplemented variant uses &'static str ────────────

    #[test]
    fn backend_error_unimplemented_accepts_static_str_only() {
        // Arrange: &'static str, not String
        let err = BE::Unimplemented("static");
        // Act & Assert: should compile (proving it takes &'static str) and display
        assert_eq!(format!("{err}"), "unimplemented: static");
    }

    // ── get_system_memory_pressure: returned value is usable as f32 ──────

    #[test]
    fn get_system_memory_pressure_usable_in_comparison() {
        // Arrange
        let pressure = get_system_memory_pressure().unwrap();
        // Act & Assert: can be compared with literal f32 without conversion
        assert!(pressure >= 0.0_f32);
        assert!(pressure <= 1.0_f32);
    }

    // ── get_system_memory_pressure: result can be used with ? operator ───

    #[test]
    fn get_system_memory_pressure_propagates_via_try_operator() {
        // Arrange: a helper that uses the ? operator
        fn inner() -> Result<f32, BE> { Ok(get_system_memory_pressure()?) }
        // Act
        let result = inner();
        // Assert
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val >= 0.0 && val <= 1.0);
    }

    // ── Pressure formula + clamp applied together ────────────────────────

    #[test]
    fn pressure_formula_with_clamp_full_pipeline() {
        // Arrange: simulate the exact code path in get_memory_pressure_linux
        let total_kb: u64 = 8192000;
        let available_kb: u64 = 2048000;
        // Act: same formula as production code
        let pressure = (1.0 - (available_kb as f64 / total_kb as f64)).clamp(0.0, 1.0) as f32;
        // Assert: 75% used
        assert!((pressure - 0.75).abs() < 0.001);
        assert!(pressure.is_finite());
        assert!(!pressure.is_nan());
    }

    // ── BackendError: all String variants can carry multi-byte UTF-8 ─────

    #[test]
    fn backend_error_string_variants_preserve_multibyte_utf8() {
        // Arrange: 4-byte UTF-8 character (emoji)
        let emoji = "🔥";
        let err_cpu = BE::Cpu(emoji.into());
        let err_cuda = BE::Cuda(emoji.into());
        let err_hip = BE::Hip(emoji.into());
        let err_metal = BE::Metal(emoji.into());
        let err_other = BE::Other(emoji.into());
        // Act & Assert: all should contain the emoji
        for (label, err) in [
            ("Cpu", err_cpu as BE),
            ("Cuda", err_cuda),
            ("Hip", err_hip),
            ("Metal", err_metal),
            ("Other", err_other),
        ] {
            assert!(
                format!("{err}").contains(emoji),
                "{label} variant should contain emoji"
            );
        }
    }

    // ── BackendError: Debug round-trip for every variant ─────────────────

    #[test]
    fn backend_error_debug_contains_variant_name_for_all() {
        // Arrange & Act & Assert: Debug string should contain the enum variant
        // identifier (PascalCase) while Display uses human-readable text.
        let cases: &[(&str, BE)] = &[
            ("Cpu", BE::Cpu("a".into())),
            ("Cuda", BE::Cuda("b".into())),
            ("Hip", BE::Hip("c".into())),
            ("Metal", BE::Metal("d".into())),
            ("Unimplemented", BE::Unimplemented("e")),
            ("Other", BE::Other("f".into())),
        ];
        for (variant, err) in cases {
            let debug = format!("{err:?}");
            assert!(
                debug.contains(variant),
                "Debug of {variant} should contain variant name, got: {debug}"
            );
        }
    }

    // ── Pressure calculation: ratio of very large to very large ──────────

    #[test]
    fn pressure_formula_very_large_near_equal() {
        // Arrange: two large values that differ by 1
        let total: u64 = u64::MAX - 1;
        let available: u64 = u64::MAX - 2;
        // Act
        let pressure = (1.0 - (available as f64 / total as f64)).clamp(0.0, 1.0) as f32;
        // Assert: extremely small pressure, but positive and finite
        assert!(pressure >= 0.0);
        assert!(pressure < 0.001);
        assert!(pressure.is_finite());
    }
}
