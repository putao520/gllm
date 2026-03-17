use crate::engine::executor::BackendError as BE;

// ---------------------------------------------------------------------------
// System memory pressure monitoring (no external crates)
// ---------------------------------------------------------------------------

pub(super) fn get_system_memory_pressure() -> Result<f32, BE> {
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
    s.trim().trim_end_matches('.').parse().unwrap_or(0)
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
