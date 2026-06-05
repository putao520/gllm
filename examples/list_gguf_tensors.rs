// Minimal GGUF tensor lister
// Reads GGUF file and prints tensor names with their shapes

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

fn skip_value(data: &[u8], pos: &mut usize, vtype: u32) -> io::Result<()> {
    match vtype {
        0..=7 => {
            // UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL
            let _ = take_slice(data, pos, 8)?;
        }
        8 => {
            // STRING
            let str_len = usize::try_from(read_u64(data, pos)?).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "string length overflow")
            })?;
            let _ = take_slice(data, pos, str_len)?;
        }
        9 => {
            // ARRAY
            let arr_len = read_u64(data, pos)?;
            let arr_type = read_u32(data, pos)?;
            for _ in 0..arr_len {
                skip_value(data, pos, arr_type)?;
            }
        }
        10 | 11 => {
            // UINT64, INT64
            let _ = take_slice(data, pos, 8)?;
        }
        12 => {
            // FLOAT64
            let _ = take_slice(data, pos, 16)?;
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown type {vtype}"),
            )
            .into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let program = args
        .next()
        .map(|p| Path::new(&p).display().to_string())
        .unwrap_or_else(|| "list_gguf_tensors".to_string());
    let Some(path) = args.next() else {
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

    // Skip magic, version, tensor_count (already at 32 after header)
    let mut pos = 32usize;

    // Skip KV pairs
    let kv_count = read_u64(&data, &mut pos)?;
    for _ in 0..kv_count {
        let key_len = usize::try_from(read_u64(&data, &mut pos)?)?;
        let _ = take_slice(&data, &mut pos, key_len)?;
        let vtype = read_u32(&data, &mut pos)?;
        skip_value(&data, &mut pos, vtype)?;
    }

    // Read tensors
    let tensor_count = read_u64(&data, &mut pos)?;
    println!("Tensor count: {tensor_count}\n");

    for i in 0..tensor_count {
        let name_len = usize::try_from(read_u64(&data, &mut pos)?)?;
        let name_bytes = take_slice(&data, &mut pos, name_len)?;
        let name = std::str::from_utf8(name_bytes).unwrap_or("invalid");

        let n_dims = read_u32(&data, &mut pos)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u64(&data, &mut pos)?);
        }

        let _dtype = read_u32(&data, &mut pos)?;
        let _offset = read_u64(&data, &mut pos)?;

        let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        println!("{i:3}. {name:60} [{}]", shape_str.join(", "));
    }

    Ok(())
}
