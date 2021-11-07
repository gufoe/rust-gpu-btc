
extern crate byteorder;
use self::byteorder::{ReadBytesExt, BigEndian, LittleEndian};
use std::io::Cursor;
use std::fs::File;
use std::io::Read;
use util;

macro_rules! load_data {
    ( $file:expr ) => {
        {
            // Load training data
            let size = util::fl_size(&$file);
            println!("Loading file with size: {0}", size);
            let mut file = File::open(&$file).unwrap();
            let mut buf = vec![0u8; size];
            file.read(&mut buf).unwrap();
            buf
        }
    };
}


pub fn load_test(data: &mut Vec<f32>, labels: &mut Vec<f32>, data_size: &mut usize, labels_size: &mut usize) {
    let raw_data = load_data!("data/btc");

    let mut rdr = Cursor::new(raw_data);
    let count_tx = rdr.read_u32::<LittleEndian>().unwrap();
    let count_rows = rdr.read_u32::<LittleEndian>().unwrap();


    let mut history = Vec::with_capacity(count_rows as usize);

    for i in 0..count_rows as usize {
        history.push(vec![0.; count_tx as usize]);
        for _j in 0..count_tx {
            history[i].push(rdr.read_f32::<LittleEndian>().unwrap());
        }
    }

    for k in 1..history.len() {
        let i = history.len()-k;
        for j in 0..history[i].len() {
            if history[i-1][j] == 0. {
                history[i][j] = 0.
            } else {
                history[i][j] = (history[i-1][j]-history[i][j])/history[i-1][j];
            }
            if history[i][j] > 1. {
                println!("wtf {0}", history[i][j]);
            }
        }
    }

    let skip = 3;

    *data_size = (count_tx*skip as u32) as usize;
    *labels_size = count_tx as usize;

    for i in skip..count_rows as usize {
        for j in 0..count_tx as usize {
            for k in 0..skip {
                data.push(history[i-skip+k][j]);
            }
            labels.push(history[i][j]);
        }
    }

    println!("translations: {0}", count_tx);
    println!("moments: {0}", count_rows);
    println!("len {0}", data.len());
}

#[allow(dead_code)]
pub fn load_test1(data: &mut Vec<f32>, labels: &mut Vec<f32>, data_size: &mut usize, labels_size: &mut usize) {
    // Load training data
    let raw_data = load_data!("data/train-images-idx3-ubyte/data");
    let raw_labels = load_data!("data/train-labels-idx1-ubyte/data");


    let mut rdr = Cursor::new(raw_data);

    rdr.read_u32::<BigEndian>().unwrap();
    let samples = rdr.read_u32::<BigEndian>().unwrap() as usize;
    let row_count = rdr.read_u32::<BigEndian>().unwrap() as usize;
    let col_count = rdr.read_u32::<BigEndian>().unwrap() as usize;
    *data_size = row_count*col_count;
    println!("Loading tests...");
    for _i in 0..*data_size*samples {
        data.push(rdr.read_u8().unwrap() as f32 / 255.);
    }


    println!("Loading labels...");
    rdr = Cursor::new(raw_labels);
    *labels_size = 10;
    rdr.read_u32::<BigEndian>().unwrap();
    assert!(samples == rdr.read_u32::<BigEndian>().unwrap() as usize);
    for _i in 0..samples as usize {
        let label = rdr.read_u8().unwrap() as usize;
        for j in 0..10 {
            labels.push(if label == j {1.} else {0.});
        }
    }
}

#[allow(dead_code)]
pub fn load_test3(data: &mut Vec<f32>, labels: &mut Vec<f32>, data_size: &mut usize, labels_size: &mut usize) {
    // Load training data
    let raw_data = load_data!("data/x_train_digits");


    let mut rdr = Cursor::new(raw_data);

    rdr.read_u32::<BigEndian>().unwrap();
    let samples = rdr.read_u32::<BigEndian>().unwrap() as usize;
    let row_count = rdr.read_u32::<BigEndian>().unwrap() as usize;
    let col_count = rdr.read_u32::<BigEndian>().unwrap() as usize;
    *data_size = row_count*col_count;
    *labels_size = row_count*col_count;
    println!("Loading tests...");
    for _i in 0..*data_size*samples {
        let val = rdr.read_u8().unwrap() as f32 / 255.;
        data.push(val);
        labels.push(val);
    }

}
