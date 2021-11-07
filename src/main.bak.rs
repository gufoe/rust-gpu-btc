mod nn;

use nn::Mutable;
use std::cmp::Ordering;
use std::fs;
#[allow(unused_imports)]
use std::io::{stdout,Write};
use std::fs::File;
use std::io::Read;
use std::io::Cursor;
use std::path::Path;

extern crate byteorder;
use byteorder::{ReadBytesExt, BigEndian};

#[macro_use]
extern crate serde_derive;
extern crate bincode;
use bincode::{serialize_into, deserialize_from};



#[allow(dead_code)]
fn test_xor(learner: &mut Learner) {
    learner.fitness = 0.;
    // println!("---");
    for _i in 0..4 {

        let a = _i%2;
        let b = if _i > 1 {1} else {0};
        let c = if a != b {1} else {0};
        // println!("{0} {1} {2}", a, b, c);
        let res = learner.network.process(&mut vec!(
            a as f32, b as f32
        ));
        learner.fitness+= -(c as f32-res[0]).abs()
    }
}


#[allow(dead_code)]
fn test(learner: &mut Learner, data: &[Vec<f32>], labels: &[Vec<f32>]) -> f32 {
    learner.fitness = 0.;
    for i in 0..data.len() {
        let res = learner.network.process(&data[i]);
        //
        // if i == 2 {
        //     println!("\n--> {}\n", res[5]);
        // }

        // let mut max = -1 as i32;
        for j in 0..res.len() {
            // if max < 0 || res[max as usize] < res[j] {
            //     max = j as i32;
            // }
            // println!("-> len {}", res[j]);
            learner.fitness-= (res[j]-labels[i][j]).abs()
        }

        // let answer = if res[j] > 1. {1.} else if res[j] < 0. {0.} else {res[j]};
        // let answer = res[j];
        // learner.fitness+= labels[i][max as usize]

    }
    learner.fitness
}

fn fl_size(path: &str) -> usize {
    let metadata = fs::metadata(path).unwrap();
    metadata.len() as usize
}


// #[allow(dead_code)]
// fn test_btc() {
//
//
//     // Load training data
//     let data_file = "../cry/cache.bin";
//     let size = fl_size(data_file);
//     let mut file=File::open(data_file).unwrap();
//     let mut buf = vec![0u8; size as usize];
//     file.read(&mut buf).unwrap();
//     let mut rdr = Cursor::new(buf);
//
//     let count_tx = rdr.read_u32::<LittleEndian>().unwrap();
//     let count_rows = rdr.read_u32::<LittleEndian>().unwrap();
//
//     let mut data = Vec::with_capacity(count_rows as usize);
//
//     for i in 0..count_rows as usize {
//         data.push(Vec::with_capacity(count_tx as usize));
//         for _j in 0..count_tx {
//             data[i].push(rdr.read_f32::<LittleEndian>().unwrap());
//         }
//     }
//
//     for k in 1..data.len() {
//         let i = data.len()-k;
//         for j in 0..data[i].len() {
//             if data[i-1][j] == 0. {
//                 data[i][j] = 0.
//             } else {
//                 data[i][j] = (data[i-1][j]-data[i][j])/data[i-1][j];
//             }
//             if data[i][j] > 1. {
//                 println!("wtf {0}", data[i][j]);
//             }
//         }
//     }
//     println!("translations: {0}", count_tx);
//     println!("moments: {0}", count_rows);
//     println!("len {0}", data.len());
//     let a = 1 as usize;
//     let b = 100 as usize;
//     let mut total_error = 0.;
//     {
//         for i in a..b {
//             for j in &data[i] {
//                 total_error+= j.abs()
//             }
//         }
//         println!("total error = {0}", total_error);
//     }
//
//
//
//
//
//
//
//
//     // Create population
//     let mut pop: Vec<Learner> = vec!();
//     for _i in 0..100 {
//         // let mut n = nn::Network::new(&[2, 4, 2]);
//         let mut n = nn::Network::new(&[count_tx as usize, 30, count_tx as usize]);
//         for l in &mut n.layers {
//             l.mrate = 0.001;
//         }
//         n.mutate();
//         pop.push(Learner { network: n, fitness: 0. });
//     }
//
//
//     // Start training
//     #[allow(unused_assignments)]
//     let mut fit: f32 = -10000000000.;
//     let mut generation = 0;
//     loop {
//         // let a = nn::rand_int(0, data.len() as i32 - 1000) as usize;
//         for i in 0..pop.len() {
//             // println!("Testing {0}", i);
//             test(&mut pop[i], &data, &data, a, b);
//             // test_xor(&mut pop[i]);
//         }
//         let mut tmp: Vec<Learner> = Vec::new();
//         nn::ga(&mut pop, &mut tmp);
//
//         fit = pop[0].fitness/total_error;
//         let mut avg: f32 = 0.;
//         for c in &pop {
//             avg+= c.fitness/total_error;
//         }
//         avg/= pop.len() as f32;
//
//         // if pop[0].fitness > fit {
//             // for l in &pop[0].network.layers {
//             //     print!("{0}, ", l.mrate);
//             // }
//             // for i in &pop[0].network.layers[0].weights {
//             //     print!("{0}, ", i);
//             // }
//             // println!("");
//             println!("{0}\t{1}\t{2}", generation, fit, avg);
//         // }
//         pop = tmp;
//         generation+= 1;
//     }
// }

#[allow(unused_macros)]
macro_rules! load_data {
    ( $file:expr ) => {
        {
            // Load training data
            let size = fl_size(&$file);
            println!("Loading file with size: {0}", size);
            let mut file=File::open(&$file).unwrap();
            let mut buf = vec![0u8; size];
            file.read(&mut buf).unwrap();
            buf
        }
    };
}

#[allow(unused_macros)]
macro_rules! mb {
    ( $file:expr ) => {
        {
            // Load training data
            format!("{:.*}", 2, fl_size(&$file) as f32 / (1024.*1024.))
        }
    };
}



mod calc;

extern crate flate2;

use flate2::write::ZlibEncoder;
use flate2::read::ZlibDecoder;
use flate2::Compression;

fn main() {



    let mut n = nn::Network::new(&[2, 3, 1]);
    for l in &mut n.layers {
        l.mrate = 1.;
    }
    n.mutate();
    n.mutate();
    n.mutate();
    println!("c: {}\nw: {:?}", n.weight_count(), n.flat_weights());


    let mut tester = calc::Tester::new(4);
    let data = vec![0., 0., 0., 1., 1., 0., 1., 1.];
    let labels = vec![0., 0., 0., 0.];

    tester.set_test(&data, &labels);
    let gpu = tester.process(&n);
    let cpu = vec![
        n.process(&data[0..2])[0],
        n.process(&data[2..4])[0],
        n.process(&data[4..6])[0],
        n.process(&data[6..8])[0]
    ];

    println!("gpu: {:?}", gpu);
    println!("cpu: {:?}", &cpu);

    n.mutate();

    let gpu = tester.process(&n);
    let cpu = vec![
        n.process(&data[0..2])[0],
        n.process(&data[2..4])[0],
        n.process(&data[4..6])[0],
        n.process(&data[6..8])[0]
    ];

    println!("gpu: {:?}", gpu);
    println!("cpu: {:?}", &cpu);

    return;
    //
    //
    // // Load training data
    // let raw_data = load_data!("../../Downloads/train-images-idx3-ubyte/data");
    // let raw_labels = load_data!("../../Downloads/train-labels-idx1-ubyte/data");
    //
    //
    // let mut rdr = Cursor::new(raw_data);
    //
    // rdr.read_u32::<BigEndian>().unwrap();
    // let samples = rdr.read_u32::<BigEndian>().unwrap();
    // let row_count = rdr.read_u32::<BigEndian>().unwrap();
    // let col_count = rdr.read_u32::<BigEndian>().unwrap();
    // let mut data = Vec::with_capacity(samples as usize);
    // println!("Loading tests...");
    // for i in 0..samples as usize {
    //     // println!("{0} size: {1}", i, row_count*col_count);
    //     data.push(Vec::with_capacity((row_count*col_count) as usize));
    //     for _j in 0..row_count*col_count {
    //         data[i].push(rdr.read_u8().unwrap() as f32 / 255.);
    //     }
    // }
    //
    //
    // println!("Loading labels...");
    // rdr = Cursor::new(raw_labels);
    //
    // rdr.read_u32::<BigEndian>().unwrap();
    // let samples = rdr.read_u32::<BigEndian>().unwrap();
    // let mut labels: Vec<Vec<f32>> = Vec::with_capacity(samples as usize);
    // for i in 0..samples as usize {
    //     labels.push(Vec::with_capacity(10));
    //     let label = rdr.read_u8().unwrap() as usize;
    //     for j in 0..10 {
    //         labels[i].push(if label == j {1.} else {0.});
    //     }
    // }
    //
    //
    //
    // let input_c = data[0].len();
    // let output_c = labels[0].len();
    // println!("sets: {0} -> {1}", data.len(), labels.len());
    // println!("inputs: {0}", input_c);
    // println!("outputs: {0}", output_c);
    //
    // // println!("inputs: {:?}", data[0]);
    // // println!("inputs: {:?}", labels[0]);
    //
    //
    //
    // let status_file = "/dev/shm/status.bin.gz";
    //
    //
    //
    // // Create population
    // let mut pop: Vec<Learner> = Vec::new();
    // if !Path::new(status_file).exists() {
    //     print!("Creating a new population...");
    //     stdout().flush().unwrap();
    //     for _i in 0..1 {
    //         // let mut n = nn::Network::new(&[2, 4, 2]);
    //         let mut n = nn::Network::new(&[input_c as u32, 30, 2, output_c as u32]);
    //         for l in &mut n.layers {
    //             l.mrate = 10./(input_c * 30 + 30 * output_c) as f32;
    //         }
    //         for _j in 0..1999 {
    //             n.mutate();
    //         }
    //         pop.push(Learner { network: n, fitness: 0. });
    //     }
    //     println!("done (len: {0})", pop.len());
    // } else {
    //     // Load the old population
    //     print!("Loading population from file...");
    //     stdout().flush().unwrap();
    //     let file = File::open(status_file).unwrap();
    //     let mut decoder = ZlibDecoder::new(file);
    //     pop = deserialize_from(&mut decoder, bincode::Infinite).unwrap();
    //     println!(" (size: {0} MB) (len: {1})", mb!(status_file), pop.len());
    //
    //     for l in &mut pop {
    //         for l in &mut l.network.layers {
    //             l.mrate = 1000./(input_c * 30 + 30 * output_c) as f32;
    //         }
    //     }
    // }
    //
    // let slice_size = 512;
    // let mut total_error;
    // let mut tester = calc::Tester::new(slice_size);
    //
    //
    // let data_flat = calc::flatify(&data);
    // let labels_flat = calc::flatify(&labels);
    //
    //
    // // Start training
    // #[allow(unused_assignments)]
    // let mut fit: f32 = -10000000000.;
    // let mut generation = 0;
    // let mut a = 0;
    // loop {
    //     a+= slice_size;
    //     while a+slice_size >= data.len() as usize {
    //         a-= samples as usize;
    //     }
    //
    //     total_error = 0.;
    //     for i in &labels_flat[a*output_c..(a+slice_size)*output_c] {
    //         total_error+= i
    //     }
    //
    //     let test_data = &data_flat[a*input_c..(a+slice_size)*input_c];
    //     let test_labels = &labels_flat[a*output_c..(a+slice_size)*output_c];
    //     tester.set_test(test_data, test_labels);
    //
    //
    //     // let a = nn::rand_int(0, data.len() as i32 - 1000) as usize;
    //     for i in 0..pop.len() {
    //         // if i%10 == 0 {
    //
    //             // print!("\r{0}%", format!("{:.*}", 2, 100. * i as f32 /pop.len() as f32));
    //             // stdout().flush().unwrap();
    //         // }
    //         // println!("testing {0}", i);
    //         let res = tester.process(&pop[i].network);
    //         println!("res -data:\n{:?}", &res[0..pop[i].network.layers[0].output.len()]);
    //         println!("\n\n\ntest-data:\n{:?}\n\n\n", pop[i].network.layers[0].output);
    //         pop[i].fitness = res[0];
    //         // pop[i].fitness = test(&mut pop[i], &data[a..a+slice_size], &labels[a..a+slice_size]);
    //         // test_xor(&mut pop[i]);
    //     }
    //     // print!("\t");
    //             fit = pop[0].fitness;
    //     let mut tmp: Vec<Learner> = Vec::new();
    //     nn::ga(&mut pop, &mut tmp);
    //
    //     let mut avg: f32 = 0.;
    //     for c in &pop {
    //         avg+= c.fitness/total_error;
    //     }
    //     avg/= pop.len() as f32;
    //
    //     print!("{}\t{}\t{}\t{}\t{}\t{}\t", generation, a, a+slice_size, fit, avg, test(&mut pop[0], &data[a..a+slice_size], &labels[a..a+slice_size]));
    //     stdout().flush().unwrap();
    //
    //     pop = tmp;
    //
    //     // Save the new population
    //     if (generation+1) % 20 == 0 {
    //         if Path::new(status_file).exists() {
    //             fs::rename(status_file, status_file.to_owned()+".bak").unwrap();
    //         }
    //         print!("saving...");
    //         stdout().flush().unwrap();
    //         let file = File::create(status_file).unwrap();
    //         let mut encoder = ZlibEncoder::new(file, Compression::default());
    //         serialize_into(&mut encoder, &pop, bincode::Infinite).unwrap();
    //         print!("{0} MB", mb!(status_file));
    //         stdout().flush().unwrap();
    //     }
    //
    //     println!();
    //     generation+= 1;
    // }


}
