extern crate ocl;
use self::ocl::{ProQue, Buffer, MemFlags, Kernel};
use std::fs::File;
use std::io::prelude::*;


use nn;

#[allow(dead_code)]
pub fn flatify(data: &[Vec<f32>]) -> Vec<f32> {
    let mut data_flat: Vec<f32> = Vec::with_capacity(data.len() * data[0].len());
    for i in data {
        for j in i {
            data_flat.push(*j)
        }
    }
    data_flat
}

pub struct Tester {
    pub dim: usize,
    kernel: Kernel,
    queue: ProQue,
    fit_buf: Buffer<f32>,
    res: Vec<f32>,
}

impl Tester {
    pub fn new(dim: usize) -> Tester {
        // Prepare kernel source
        // println!("Prepare kernel source...");
        let mut file = File::open("exec.cl").unwrap();
        let mut src = String::new();
        file.read_to_string(&mut src).unwrap();

        // Create queue
        // println!("Create queue...");
        let queue = ProQue::builder()
            .src(src)
            .dims(dim)
            .build().expect("Build ProQue");


        let u_buf: Buffer<u32> = queue.create_buffer().unwrap();
        let f1_buf: Buffer<f32> = queue.create_buffer().unwrap();
        let f2_buf: Buffer<f32> = queue.create_buffer().unwrap();
        let f3_buf: Buffer<f32> = queue.create_buffer().unwrap();

        let fit = vec![0f32; dim];

        let fit_buf = Buffer::builder()
            .queue(queue.queue().clone())
            .flags(MemFlags::new().read_write().use_host_ptr())
            .dims(fit.len())
            .host_data(&fit)
            .build().unwrap();

        // Create the kernel
        // println!("Create the kernel...");
        let kernel = queue.create_kernel("process").unwrap()
            .arg_scl_named("layer_c", Some(0 as u32))
            .arg_buf_named("net", Some(u_buf))
            .arg_buf_named("weights", Some(f1_buf))
            .arg_buf_named("data", Some(f2_buf))
            .arg_buf_named("labels", Some(f3_buf))
            .arg_buf(&fit_buf);

        Tester { dim, kernel, queue, fit_buf, res: fit }
    }

    pub fn set_test(&mut self, data: &[f32], labels: &[f32]) {

        let data_buf = Buffer::builder()
            .queue(self.queue.queue().clone())
            .flags(MemFlags::new().read_only().use_host_ptr())
            .dims(data.len())
            .host_data(&data)
            .build().unwrap();

        let labels_buf = Buffer::builder()
            .queue(self.queue.queue().clone())
            .flags(MemFlags::new().read_only().use_host_ptr())
            .dims(labels.len())
            .host_data(&labels)
            .build().unwrap();

        self.kernel.set_arg_buf_named("data", Some(data_buf)).unwrap();
        self.kernel.set_arg_buf_named("labels", Some(labels_buf)).unwrap();
    }

    pub fn process(&mut self, network: &nn::Network) -> f32 {

        // Prepare data
        let weights = network.flat_weights();
        let net: Vec<u32> = network.definition.clone();



        // Prepare buffers
        let net_buf = Buffer::builder()
            .queue(self.queue.queue().clone())
            .flags(MemFlags::new().read_only().use_host_ptr())
            .dims(net.len())
            .host_data(&net)
            .build().unwrap();
        let weights_buf = Buffer::builder()
            .queue(self.queue.queue().clone())
            .flags(MemFlags::new().read_only().use_host_ptr())
            .dims(weights.len())
            .host_data(&weights)
            .build().unwrap();

        self.kernel.set_arg_scl_named("layer_c", net.len() as u32).unwrap();
        self.kernel.set_arg_buf_named("net", Some(net_buf)).unwrap();
        self.kernel.set_arg_buf_named("weights", Some(weights_buf)).unwrap();


        // println!("Running the self.kernel...");
        unsafe { self.kernel.enq().unwrap(); }

        let mut fit = self.res.clone();//vec!(0f32; self.dim);
        // println!("Reading fitness...");
        self.fit_buf.read(&mut fit).enq().unwrap();
        // println!("Done");
        let mut sum = 0.;
        for x in fit {
            sum+= x;
        }

        let l = self.res.len();
        // println!("done {}", fit[0]);
        sum/l as f32
    }
}
