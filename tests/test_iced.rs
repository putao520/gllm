use iced_x86::code_asm::*;

#[test]
fn test_iced_mov() {
    let mut asm = CodeAssembler::new(64).unwrap();
    asm.mov(r14, qword_ptr(rsp + 48)).unwrap();
    let bytes = asm.assemble(0x1000).unwrap();
    for b in bytes {
        print!("{:02x} ", b);
    }
    println!();
}
