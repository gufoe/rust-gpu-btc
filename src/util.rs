
extern crate rand;
use self::rand::Rng;

extern crate bmp;
use self::bmp::Image;
use self::bmp::Pixel;


extern crate serde;

extern crate flate2;
use self::flate2::write::ZlibEncoder;
use self::flate2::read::ZlibDecoder;
use self::flate2::Compression;

extern crate bincode;
use self::bincode::{serialize_into, deserialize_from};

use std::fs::File;
use std::path::Path;
use std::io::{stdout,Write};

use std::fs;

pub fn rand_int(a: i32, b: i32) -> i32 {
    if a == b {
        a
    } else {
        rand::thread_rng().gen_range(a, b+1)
    }
}

#[allow(dead_code)]
pub fn rand_float(a: f32, b: f32) -> f32 {
    rand::thread_rng().gen_range::<f32>(a, b)
}


pub fn fl_size(path: &str) -> usize {
    let metadata = fs::metadata(path).unwrap();
    metadata.len() as usize
}


pub fn fl_exists(path: &str) -> bool {
    Path::new(path).exists()
}

pub fn flush() {
    stdout().flush().unwrap();
}

pub fn store<T: serde::Serialize>(path: &str, thing: T) {

    if Path::new(path).exists() {
        fs::rename(path, path.to_owned()+".bak").unwrap();
    }
    let file = File::create(path).unwrap();
    let mut encoder = ZlibEncoder::new(file, Compression::default());
    serialize_into(&mut encoder, &thing, bincode::Infinite).unwrap();
}

pub fn load<T: serde::de::DeserializeOwned>(path: &str) -> T
{
    let file = File::open(path).unwrap();
    let mut decoder = ZlibDecoder::new(file);
    deserialize_from(&mut decoder, bincode::Infinite).unwrap()
}

pub fn export_bitmap(name: &str, map: &Vec<f32>, w: usize, h: usize) {
    let mut img = Image::new(w as u32, h as u32);
    // println!("{} {} {}", map.len(), w, h);
    for i in 0..w {
        for j in 0..h {
            img.set_pixel(i as u32, j as u32, Pixel::new((map[w*j+i] * 255.0) as u8, 0, 0));
        }
    }
    let _ = img.save(name);
}
