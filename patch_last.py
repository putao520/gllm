d = "src/compat/decoder_forward.rs"

with open(d, 'r') as f:
    c = f.read()

c = c.replace("OpKind::Gemm { m: seq_len,", "OpKind::Gemm { m: seq_len.into(),")
c = c.replace("OpKind::Gemm { m: 1,", "OpKind::Gemm { m: 1.into(),")

with open(d, 'w') as f: f.write(c)

