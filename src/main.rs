#[macro_use]
extern crate serde_derive;


mod nn;
mod util;
mod calc;
mod learner;
mod loader;

use learner::Learner;
use nn::Network;
use nn::Mutable;

const DB_SIZE: usize = 300;
const MRATE: f32 = 0.02;
const CHUNK_SIZE: usize = 128;
static FILE_PATH: &'static str = "state.bin.gz";


extern crate time;

fn timestamp() -> f64 {
    let timespec = time::get_time();
    // 1459440009.113178
    let mills: f64 = timespec.sec as f64 + (timespec.nsec as f64 / 1000.0 / 1000.0 / 1000.0);
    ((mills*1000.) as usize) as f64 / 1000.
}


#[allow(unused_macros)]
macro_rules! MB {
    ( $file:expr ) => {
        {
            // Load training data
            format!("{:.*}", 2, util::fl_size(&$file) as f32 / (1024.*1024.))
        }
    };
}

fn create_learner(net_shape: &[u32], mrate: f32) -> Learner {
    let mut network = Network::new(net_shape, mrate);
    network.mutate();
    Learner { network, fitness: 0. }
}


fn analyze(gen: u32, db: &Vec<Learner>) {

    let mut avg = 0.;
    for thing in db { avg+= thing.fitness }
    avg/= db.len() as f32;
    print!("{}\t{}\t{}\t{}\t", timestamp(), gen, db[0].fitness as f32, avg as f32);
    util::flush();


    // Save the new population
    if (gen+1) % 30 == 0 {

        print!("saving...");
        util::flush();

        util::store(FILE_PATH, db);

        print!("{0} MB", MB!(FILE_PATH));
        util::flush();
    }

    println!();
}


fn main() {

    // Load tests

    // let test_data: Vec<f32> = vec![0., 0., 0., 1., 1., 0., 1., 1.];
    // let test_labels: Vec<f32> = vec![0., 1., 1., 0.];
    // let test_data_size: usize = 2;
    // let test_labels_size: usize = 1;

    let mut test_data: Vec<f32> = Vec::new();
    let mut test_labels: Vec<f32> = Vec::new();
    let mut test_data_size: usize = 0;
    let mut test_labels_size: usize = 0;
    // loader::load_test(&mut test_data, &mut test_labels, &mut test_data_size, &mut test_labels_size);
    loader::load_test3(&mut test_data, &mut test_labels, &mut test_data_size, &mut test_labels_size);

    let test_samples = test_data.len()/test_data_size;

    let i = util::rand_int(0, (test_samples - CHUNK_SIZE) as i32) as usize;
    println!("data ({}): {:?}", test_data_size, &test_data[i*test_data_size..(i+1)*test_data_size]);
    println!("labels ({}): {:?}", test_labels_size, &test_labels[i*test_labels_size..(i+1)*test_labels_size]);

    // Init db
    let mut db: Vec<Learner>;
    if util::fl_exists(FILE_PATH) {
        db = util::load(FILE_PATH);
        println!("Loaded checkpoint (size: {0} MB) (len: {1})", MB!(FILE_PATH), db.len());
    } else {
        println!("Creating population...");
        db = Vec::new();
        for _i in 0..DB_SIZE {
            db.push(create_learner(&[test_data_size as u32, 4, 4, 4, test_labels_size as u32], MRATE))
        }
    }



    // Init gpu
    print!("Initializing GPU...");
    util::flush();
    let mut gpu = calc::Tester::new(CHUNK_SIZE);
    println!("done");


    println!("CPU test 1: {:?}", db[0].network.fitness(&test_data[i*test_data_size..(i+CHUNK_SIZE)*test_data_size], &test_labels[i*test_labels_size..(i+CHUNK_SIZE)*test_labels_size]));
    gpu.set_test(&test_data[i*test_data_size..(i+CHUNK_SIZE)*test_data_size], &test_labels[i*test_labels_size..(i+CHUNK_SIZE)*test_labels_size]);
    println!("GPU test 1: {:?}", gpu.process(&db[0].network));
    let i = util::rand_int(0, (test_samples - CHUNK_SIZE) as i32) as usize;
    println!("CPU test 2: {:?}", db[0].network.fitness(&test_data[i*test_data_size..(i+CHUNK_SIZE)*test_data_size], &test_labels[i*test_labels_size..(i+CHUNK_SIZE)*test_labels_size]));
    gpu.set_test(&test_data[i*test_data_size..(i+CHUNK_SIZE)*test_data_size], &test_labels[i*test_labels_size..(i+CHUNK_SIZE)*test_labels_size]);
    println!("GPU test 2: {:?}", gpu.process(&db[0].network));

    let mut gen = 0;
    let mut chunk = 0;


    loop {

        // Update the chunk indexes
        // chunk+= gpu.dim;
        if chunk+gpu.dim > test_samples {
            chunk-= test_samples-gpu.dim
        }

        // Get test slices
        let test_data = &test_data[chunk*test_data_size..(chunk+gpu.dim)*test_data_size];
        let test_labels = &test_labels[chunk*test_labels_size..(chunk+gpu.dim)*test_labels_size];

        // Set GPU buffers
        gpu.set_test(test_data, test_labels);

        // Update fitnesses
        for learner in &mut db {
            learner.fitness = gpu.process(&learner.network);
            // learner.f    itness = learner.network.fitness(test_data, test_labels);
            // println!("{} = {}", learner.fitness, learner.network.fitness(test_data, test_labels));
        }

        // Generate next population
        {
            let mut next: Vec<Learner> = Vec::with_capacity(db.len());
            print!("ga...\t");
            util::flush();
            nn::ga(&mut db, &mut next);
            analyze(gen, &db);
            {
                let out = db[0].network.process(&test_data[0..test_data_size]);
                util::export_bitmap("out.bmp", &out, 28, 28);
            }
            db = next;
        }

        gen+= 1;
    }

}
