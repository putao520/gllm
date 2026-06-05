use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

fn take_slice<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> io::Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "position overflow"))?;
    let slice = data.get(*pos..end).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("unexpected EOF at offset {} (need {len} bytes)", *pos),
        )
    })?;
    *pos = end;
    Ok(slice)
}

fn read_u64(data: &[u8], pos: &mut usize) -> io::Result<u64> {
    let bytes = take_slice(data, pos, 8)?;
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid u64 slice"))?;
    Ok(u64::from_le_bytes(arr))
}

fn read_u32(data: &[u8], pos: &mut usize) -> io::Result<u32> {
    let bytes = take_slice(data, pos, 4)?;
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid u32 slice"))?;
    Ok(u32::from_le_bytes(arr))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let program = args
        .next()
        .map(|p| Path::new(&p).display().to_string())
        .unwrap_or_else(|| "list_gguf_keys".to_string());
    let Some(path) = args.next() else {
        // Example program: return a clear error instead of panicking.
        eprintln!("Usage: {program} <path-to-model.gguf>");
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "missing GGUF file path").into());
    };

    let mut file = File::open(&path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    if data.len() < 32 {
        return Err(
            io::Error::new(io::ErrorKind::UnexpectedEof, "GGUF header is truncated").into(),
        );
    }

    let mut header_pos = 24usize;
    let kv_count = read_u64(&data, &mut header_pos)?;
    println!("KV count: {kv_count}\n");

    let mut pos = 32usize;
    for i in 0..kv_count {
        let key_len = usize::try_from(read_u64(&data, &mut pos)?)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "key length overflow"))?;
        let key_bytes = take_slice(&data, &mut pos, key_len)?;
        let key = std::str::from_utf8(key_bytes).unwrap_or("invalid");

        let vtype = read_u32(&data, &mut pos)?;
        let type_name = match vtype {
            0 => "UINT8",
            1 => "INT8",
            2 => "UINT16",
            3 => "INT16",
            4 => "UINT32",
            5 => "INT32",
            6 => "FLOAT32",
            7 => "BOOL",
            8 => "STRING",
            9 => "ARRAY",
            10 => "UINT64",
            11 => "INT64",
            12 => "FLOAT64",
            _ => "UNKNOWN",
        };

        if vtype == 8 {
            let str_len = usize::try_from(read_u64(&data, &mut pos)?).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "string length overflow")
            })?;
            let _ = take_slice(&data, &mut pos, str_len)?;
        } else if vtype <= 7 {
            let _ = take_slice(&data, &mut pos, 8)?;
        } else if vtype == 9 {
            let arr_len = read_u64(&data, &mut pos)?;
            let _arr_type = read_u32(&data, &mut pos)?;
            let arr_len = usize::try_from(arr_len).unwrap_or(usize::MAX / 4);
            let skip_len = arr_len.saturating_mul(4).min(400);
            let _ = take_slice(&data, &mut pos, skip_len)?;
        } else {
            let _ = take_slice(&data, &mut pos, 16)?;
        }

        println!("{i:2}. {key:50} {type_name}");
    }

    Ok(())
}
