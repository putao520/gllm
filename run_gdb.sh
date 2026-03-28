#!/bin/bash
gdb -batch \
  -ex "file /home/putao/code/rust/gllm/target/debug/deps/test_e2e_generator-66f67dd6c361d395" \
  -ex "run e2e_generator_safetensors --nocapture" \
  -ex "info registers r10 r11 r12 r13 r14 r15 rsp rbp rdi rdx rcx rbx r8 r9" \
  -ex "x/10xg \$rbp-96" \
  -ex "continue"
