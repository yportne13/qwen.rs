use std::{str::FromStr, io::Write};

use candle_core::{Error, quantized::ggml_file};
use candle_nn::VarBuilder;

fn u32_to_array_of_u8 (x:u32) -> [u8;4] {
    let b1 : u8 = ((x >> 24) & 0xff) as u8;
    let b2 : u8 = ((x >> 16) & 0xff) as u8;
    let b3 : u8 = ((x >> 8) & 0xff) as u8;
    let b4 : u8 = (x & 0xff) as u8;
    [b4, b3, b2, b1]
}

fn main() -> Result<(), Error> {
    let filenames = vec![
        std::path::PathBuf::from_str("weight/model-00001-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00002-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00003-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00004-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00005-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00006-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00007-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00008-of-00008.safetensors").unwrap(),
    ];
    let handles = filenames
        .iter()
        .map(|f| Ok(unsafe { candle_core::safetensors::MmapedFile::new(f.as_path())? }))
        .collect::<Result<Vec<_>, Error>>()?;
    let tensors: Vec<_> = handles
        .iter()
        .map(|h| h.deserialize())
        .collect::<Result<Vec<_>, Error>>()?;

    let device = candle_core::Device::Cpu;
    let vb = VarBuilder::from_safetensors(tensors, candle_core::DType::F16, &device);

    let mut ggml = std::fs::File::create("qwen-7b-ggml.bin")?;
    //ggml.write_all(b"tjgg")?;
    //ggml.write_all(&[3, 0, 0, 0])?;
    ggml.write_all(b"lmgg")?;

    ggml.write_all(&u32_to_array_of_u8(151936))?;//vocab_size
    ggml.write_all(&u32_to_array_of_u8(4096))?;//hidden_size
    ggml.write_all(&[0, 0, 0, 0])?;
    ggml.write_all(&[32, 0, 0, 0])?;
    ggml.write_all(&[32, 0, 0, 0])?;
    ggml.write_all(&[0, 0, 0, 0])?;
    ggml.write_all(&[0, 0, 0, 0])?;

    for _ in 0..151936 {
        ggml.write_all(&[0, 0, 0, 0])?;
        ggml.write_all(&[0, 0, 0, 0])?;
    }

    ///////
    fn write_2dim(vb: &VarBuilder, file: &mut std::fs::File, size: (u32, u32), name: &str) -> Result<(), Error> {
        file.write_all(&u32_to_array_of_u8(2))?;//dims
        file.write_all(&u32_to_array_of_u8(name.len() as u32))?;//num_len
        file.write_all(&[1, 0, 0, 0])?;//dtype F16
        //file.write_all(&[2, 0, 0, 0])?;//dtype q4_0
        //file.write_all(&[6, 0, 0, 0])?;//dtype q5_0
        //file.write_all(&[8, 0, 0, 0])?;//dtype q8_0
        file.write_all(&u32_to_array_of_u8(size.1))?;
        file.write_all(&u32_to_array_of_u8(size.0))?;
        file.write_all(name.as_bytes())?;
        /*let tensor = vb.get((size.0 as usize, size.1 as usize), name)?
            .to_dtype(candle_core::DType::F16)?;//: Vec<Vec<half::f16>>
            //.to_vec2()?;
        use candle_core::quantized::GgmlType;
        type T = candle_core::quantized::k_quants::BlockQ5_0;
        let src = tensor
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut data = vec![T::zeros(); src.len() / T::BLCK_SIZE];
        T::from_float(&src, &mut data)?;
        unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
            ::core::slice::from_raw_parts(
                (p as *const T) as *const u8,
                ::core::mem::size_of::<T>(),
            )
        }
        let to_write = data.into_iter()
            .flat_map(|x| unsafe{any_as_u8_slice(&x)}.to_vec())
            .collect::<Vec<_>>();
        file.write_all(&to_write)?;*/
        let tensor: Vec<Vec<half::f16>> = vb.get((size.0 as usize, size.1 as usize), name)?
            .to_dtype(candle_core::DType::F16)?
            .to_vec2()?;
        for t1 in tensor {
            file.write_all(&t1.into_iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())?;
        }
        Ok(())
    }
    fn write_1dim(vb: &VarBuilder, file: &mut std::fs::File, size: u32, name: &str) -> Result<(), Error> {
        file.write_all(&u32_to_array_of_u8(1))?;//dims
        file.write_all(&u32_to_array_of_u8(name.len() as u32))?;//num_len
        file.write_all(&[1, 0, 0, 0])?;//dtype: F16
        file.write_all(&u32_to_array_of_u8(size))?;
        file.write_all(name.as_bytes())?;
        let tensor: Vec<half::f16> = vb.get(size as usize, name)?
            .to_dtype(candle_core::DType::F16)?
            .to_vec1()?;
        for t in tensor {
            file.write_all(&t.to_le_bytes())?;
        }
        Ok(())
    }
    
    write_2dim(&vb, &mut ggml, (151936, 4096), "transformer.wte.weight")?;
    for l_idx in 0..32 {
        write_2dim(&vb, &mut ggml, (4096*3, 4096), &format!("transformer.h.{l_idx}.attn.c_attn.weight"))?;
        write_1dim(&vb, &mut ggml, 4096*3, &format!("transformer.h.{l_idx}.attn.c_attn.bias"))?;
        write_2dim(&vb, &mut ggml, (4096, 4096), &format!("transformer.h.{l_idx}.attn.c_proj.weight"))?;
        write_2dim(&vb, &mut ggml, (11008, 4096), &format!("transformer.h.{l_idx}.mlp.w1.weight"))?;
        write_2dim(&vb, &mut ggml, (11008, 4096), &format!("transformer.h.{l_idx}.mlp.w2.weight"))?;
        write_2dim(&vb, &mut ggml, (4096, 11008), &format!("transformer.h.{l_idx}.mlp.c_proj.weight"))?;
        write_1dim(&vb, &mut ggml, 4096, &format!("transformer.h.{l_idx}.ln_1.weight"))?;
        write_1dim(&vb, &mut ggml, 4096, &format!("transformer.h.{l_idx}.ln_2.weight"))?;
        println!("layer {}", l_idx);
    }
    write_1dim(&vb, &mut ggml, 4096, "transformer.ln_f.weight")?;
    write_2dim(&vb, &mut ggml, (151936, 4096), "lm_head.weight")?;
    
    ///////
    let mut ggml = std::fs::File::open("qwen-7b-ggml.bin")?;
    let model = ggml_file::Content::read(&mut ggml)?;
    println!("params: {:?}", model.hparams);
    for t in model.tensors.iter() {
        println!("{}, {:?}", t.0, t.1)
    }

    Ok(())
}
