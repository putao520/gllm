#![cfg(test)]
use gllm::{Client, ModelKind};

/// Install a SIGSEGV handler that prints the faulting RIP + fault address +
/// /proc/self/maps so RIP can be mapped to a JIT code offset (no GDB).
#[cfg(unix)]
unsafe fn install_segfault_handler() {
    extern "C" {
        fn sigaction(signum: i32, act: *const Sigaction, old: *mut Sigaction) -> i32;
    }
    #[repr(C)]
    struct Sigaction {
        sa_sigaction: usize,
        sa_mask: [u64; 16],
        sa_flags: i32,
        sa_restorer: usize,
    }
    extern "C" fn handler(_sig: i32, info: *mut u8, _uctx: *mut u8) {
        unsafe {
            let si_addr = *(info.add(16) as *const usize);
            let mctx = _uctx.add(40);
            let rip = *(mctx.add(16 * 8) as *const usize);
            // Dump key gregs from ucontext to understand the failing pointer.
            // gregs indices (x86_64): R8=0,R9=1,R10=2,R11=3,R12=4,R13=5,R14=6,R15=7,
            // RDI=8,RSI=9,RBP=10,RBX=11,RDX=12,RAX=13,RCX=14,RSP=15,RIP=16.
            let r10 = *(mctx.add(2 * 8) as *const usize);
            let r14 = *(mctx.add(6 * 8) as *const usize);
            let rbp = *(mctx.add(10 * 8) as *const usize);
            let rax = *(mctx.add(13 * 8) as *const usize);
            let rcx = *(mctx.add(14 * 8) as *const usize);
            let r8 = *(mctx.add(0 * 8) as *const usize);
            let rdx = *(mctx.add(12 * 8) as *const usize);
            let rbx = *(mctx.add(11 * 8) as *const usize);
            let rsi = *(mctx.add(9 * 8) as *const usize);
            let rdi = *(mctx.add(8 * 8) as *const usize);
            eprintln!("[SEGFAULT] fault_addr=0x{:x} RIP=0x{:x}", si_addr, rip);
            eprintln!("[SEGFAULT] r10(dst)=0x{:x} r14(v48 raw ctr?)=0x{:x} rdx(v304 fresh?)=0x{:x} rbx(v9 gen?)=0x{:x}", r10, r14, rdx, rbx);
            eprintln!("[SEGFAULT] rax(src)=0x{:x} rcx=0x{:x} r8=0x{:x} rsi=0x{:x} rdi=0x{:x}", rax, rcx, r8, rsi, rdi);
            // Read [rbp-0xc8] (kv_cache_ptr base) and [rbp-0xa8] (weight_base).
            let kv_cache_base = if rbp > 0xc8 { *((rbp - 0xc8) as *const usize) } else { 0 };
            let weight_base = if rbp > 0xa8 { *((rbp - 0xa8) as *const usize) } else { 0 };
            eprintln!("[SEGFAULT] [rbp-0xc8]=kv_cache_ptr? =0x{:x}  [rbp-0xa8]=weight_base? =0x{:x}", kv_cache_base, weight_base);
            // Find the r-xp mapping containing RIP, print it + RIP offset, dump bytes.
            let mut code_base: usize = 0;
            let mut code_end: usize = 0;
            if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
                for line in maps.lines() {
                    if line.contains("rwx") || line.contains("rw-p") {
                        eprintln!("[SEGFAULT][maps-rw] {}", line);
                    }
                    // executable mapping: r-xp or rwxp
                    if (line.contains("r-xp") || line.contains("rwxp")) && code_base == 0 {
                        // parse "start-end perms"
                        if let Some(dash) = line.find('-') {
                            let perms_part = &line[dash..];
                            if let Some(sp) = perms_part.find(' ') {
                                let end_str = &perms_part[1..sp];
                                if let Ok(end) = usize::from_str_radix(end_str, 16) {
                                    let start_str = &line[..dash];
                                    if let Ok(start) = usize::from_str_radix(start_str, 16) {
                                        if rip >= start && rip < end {
                                            code_base = start;
                                            code_end = end;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if code_base != 0 {
                let rip_off = rip - code_base;
                eprintln!("[SEGFAULT] code_map=0x{:x}-0x{:x} RIP_offset=0x{:x}", code_base, code_end, rip_off);
                // dump 256 bytes before RIP and 16 after RIP
                let dump_start = if rip > 256 { rip - 256 } else { code_base };
                let dump_end = if rip + 16 <= code_end { rip + 16 } else { code_end };
                eprint!("[SEGFAULT][bytes @0x{:x}-0x{:x}]:", dump_start, dump_end);
                let mut p = dump_start;
                while p < dump_end {
                    let b = unsafe { *(p as *const u8) };
                    eprint!(" {:02x}", b);
                    if p == rip - 1 { eprint!("|"); }
                    p += 1;
                }
                eprintln!();
                // Also dump a wider window at the faulting store (RIP is the instr AFTER the faulting store? No — RIP = faulting instr on SIGSEGV).
            }
            // Dump more stack slots around rbp to identify vreg spills.
            if rbp > 0x400 {
                let mut i = 0x8;
                eprint!("[SEGFAULT][stack-slots]");
                while i <= 0x400 {
                    let v = unsafe { *((rbp - i) as *const usize) };
                    eprint!(" [-0x{:x}]=0x{:x}", i, v);
                    i += 0x8;
                }
                eprintln!();
            }
            // Specifically dump the slots referenced in the faulting sequence.
            for (name, off) in [("kv_cache[-0xc8]",0xc8usize), ("dst_base[-0x3c8]",0x3c8), ("src[-0x1e40]",0x1e40), ("dst[-0x1e48]",0x1e48), ("wbase[-0xa8]",0xa8), ("[-0x3c0]",0x3c0), ("[-0x3b8]",0x3b8), ("[-0xd0]",0xd0), ("[-0xd8]",0xd8)] {
                if rbp > off {
                    let v = unsafe { *((rbp - off) as *const usize) };
                    eprintln!("[SEGFAULT][slot] {} = 0x{:x}", name, v);
                }
            }
            std::process::exit(139);
        }
    }
    let act = Sigaction {
        sa_sigaction: handler as usize,
        sa_mask: [0; 16],
        sa_flags: 0x4, // SA_SIGINFO
        sa_restorer: 0,
    };
    let _ = sigaction(11, &act, std::ptr::null_mut());
}

fn dump_v(n: usize, outfile: &str) {
    #[cfg(unix)]
    unsafe { install_segfault_handler(); }
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    let v_off = sp.named_offsets.iter().find(|(n,_,_)| n == "layer.v").map(|(_,o,_)| *o).unwrap_or(1174405120);
    let vals = sp.read_dtype_aware(v_off, 8);
    let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    let nan = vals.iter().filter(|v| v.is_nan()).count();
    let _ = writeln!(f, "[N={}] layer.v |max|={:.4} NaN={}/8", n, max, nan);
    let _ = f.flush();
}
#[test]
#[ignore]
fn diag_v_n3() { let _ = std::fs::write("/tmp/v_n3.txt",""); dump_v(3, "/tmp/v_n3.txt"); }
