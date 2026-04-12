#!/usr/bin/env python3
"""
GDB Python script for debugging JIT-generated code in gllm.

Usage:
    GLLM_JIT_DEBUG=1 gdb -x scripts/gdb_jit_debug.py --args <test-binary> <test-name> --test-threads=1

What it does:
1. Configures SIGTRAP to stop (for INT3 breakpoints in JIT code)
2. Configures SIGSEGV to stop and print
3. Auto-prints key registers at every INT3 stop
4. On SIGSEGV: dumps full state + disassembly hint
5. Tracks which breakpoint (A/B/C/D) was hit based on stderr [JIT-DBG] output

Register convention (System V AMD64 ABI for CompiledLayerFn):
    rdi = activation input ptr
    rsi = weights blob base ptr
    rdx = kv_cache / bias ptr
    r8  = output ptr
    r13 = saved activation input base
    r14 = saved weights blob base
    [rbp+16] = seq_len (stack parameter)
    [rbp+24] = output ptr (stack parameter)
    [rbp+32] = scratchpad ptr (stack parameter)
"""

import gdb
import re


class JitDebugBreakpoint(gdb.Breakpoint):
    """Breakpoint on stderr JIT-DBG output to track which fusion group is being executed."""

    def stop(self):
        return False  # Just log, don't stop


def read_reg(reg):
    """Read a register value, return hex string or '<unreadable>'."""
    try:
        val = gdb.parse_and_eval(f"${reg}")
        return format(int(val), "#018x")
    except Exception:
        return "<unreadable>"


def read_mem(addr_str, size=8):
    """Read memory at address expression, return hex or '<unreadable>'."""
    try:
        val = gdb.parse_and_eval(f"*(uint64_t*)({addr_str})")
        return format(int(val), "#018x")
    except Exception:
        return "<unreadable>"


def is_null(reg):
    """Check if register value is NULL."""
    try:
        val = int(gdb.parse_and_eval(f"${reg}"))
        return val == 0
    except Exception:
        return False


def on_stop(event):
    """Handle every stop event — print register state for JIT debugging."""
    if not isinstance(event, gdb.StopEvent):
        return

    # Check if we stopped due to SIGTRAP or SIGSEGV
    try:
        sig = gdb.parse_and_eval("$_siginfo.si_signo")
        sig = int(sig)
    except Exception:
        sig = 0

    if sig == 5:  # SIGTRAP
        print("\n" + "=" * 70)
        print("  INT3 BREAKPOINT HIT IN JIT CODE")
        print("=" * 70)

        rdi = read_reg("rdi")
        rsi = read_reg("rsi")
        rdx = read_reg("rdx")
        r8 = read_reg("r8")
        r13 = read_reg("r13")
        r14 = read_reg("r14")
        rip = read_reg("rip")

        print(f"  RIP  = {rip}")
        print(f"  RDI  = {rdi}  {'*** NULL ***' if is_null('rdi') else ''}")
        print(f"  RSI  = {rsi}  {'*** NULL ***' if is_null('rsi') else ''}")
        print(f"  RDX  = {rdx}")
        print(f"  R8   = {r8}")
        print(f"  R13  = {r13}  {'*** NULL ***' if is_null('r13') else '(saved activation)'}")
        print(f"  R14  = {r14}  {'*** NULL ***' if is_null('r14') else '(saved weights)'}")

        # Stack parameters
        try:
            rbp = int(gdb.parse_and_eval("$rbp"))
            print(f"  [rbp+16] = {read_mem(f'{rbp}+16')}  (seq_len)")
            print(f"  [rbp+24] = {read_mem(f'{rbp}+24')}  (output ptr)")
            print(f"  [rbp+32] = {read_mem(f'{rbp}+32')}  (scratchpad ptr)")
        except Exception:
            pass

        print("-" * 70)
        print("  Commands:")
        print("    x/20i $rip-20     — disassemble around breakpoint")
        print("    c                  — continue to next breakpoint")
        print("    info registers     — full register dump")
        print("=" * 70 + "\n")

    elif sig == 11:  # SIGSEGV
        print("\n" + "#" * 70)
        print("  ### SIGSEGV — SEGMENTATION FAULT IN JIT CODE ###")
        print("#" * 70)

        rip = read_reg("rip")
        rdi = read_reg("rdi")
        rsi = read_reg("rsi")
        rdx = read_reg("rdx")
        rax = read_reg("rax")

        print(f"  RIP  = {rip}  (crash address)")
        print(f"  RDI  = {rdi}  {'*** NULL — likely cause ***' if is_null('rdi') else ''}")
        print(f"  RSI  = {rsi}  {'*** NULL — likely cause ***' if is_null('rsi') else ''}")
        print(f"  RDX  = {rdx}  {'*** NULL ***' if is_null('rdx') else ''}")
        print(f"  RAX  = {rax}")
        print(f"  R13  = {read_reg('r13')}  (saved activation base)")
        print(f"  R14  = {read_reg('r14')}  (saved weights base)")

        try:
            rbp = int(gdb.parse_and_eval("$rbp"))
            print(f"  [rbp+16] = {read_mem(f'{rbp}+16')}  (seq_len)")
            print(f"  [rbp+24] = {read_mem(f'{rbp}+24')}  (output ptr)")
            print(f"  [rbp+32] = {read_mem(f'{rbp}+32')}  (scratchpad ptr)")
        except Exception:
            pass

        print("-" * 70)
        print("  Diagnostic commands:")
        print("    x/40i $rip-40     — disassemble before crash")
        print("    info proc mappings — check memory regions")
        print("    x/8xg $rdi        — inspect target memory")
        print("#" * 70 + "\n")


def setup():
    """Configure GDB for JIT debugging."""
    # Signal handling: stop on SIGTRAP and SIGSEGV
    gdb.execute("handle SIGTRAP stop print pass")
    gdb.execute("handle SIGSEGV stop print pass")

    # Disable pagination so output isn't paused
    gdb.execute("set pagination off")

    # Register stop handler
    gdb.events.stop.connect(on_stop)

    print("gllm JIT debug script loaded.")
    print("  - SIGTRAP: will stop at each INT3 breakpoint")
    print("  - SIGSEGV: will stop and dump state")
    print("  - Use 'c' (continue) to proceed through breakpoints")
    print("")


setup()
