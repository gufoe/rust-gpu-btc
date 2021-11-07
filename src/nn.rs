
use util::rand_int;
use util::rand_float;

pub trait Mutable {
    fn mutate(&mut self);
    fn crossover(&mut self, &Self);
}


#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub _in : u32, // Outputs
    pub _out : u32, // Inputs
    pub mrate : f32,
    pub weights : Vec<f32>,
    pub momento: Vec<f32>,
    // biases : Vec<f32>,
    pub output : Vec<f32>,
}

impl Layer {
    fn new(inputs:u32, outputs:u32, mrate: f32) -> Layer {
        let mut l = Layer {
            _in: inputs,
            _out: outputs,
            weights: vec![0.; ((inputs*2+1)*outputs) as usize],
            momento: vec![1.; ((inputs*2+1)*outputs) as usize],
            // biases: vec![0.; outputs],
            output: vec![0.; (outputs) as usize],
            mrate: mrate,
        };

        for i in 0..outputs {
            let r = 2.;
            l.set_weight(i, 0, 0.);//rand_float(-r, r));
            for j in 0..inputs {
                l.set_weight(i, 1+j, 0.);//rand_float(-r, r));
                l.set_weight(i, 1+inputs+j, 0.);//rand_float(-r, r));
            }
        }
        return l;
    }

    // First weight for each neuron is its bias
    pub fn get_weight(&self, neuron: u32, nth: u32) -> f32 {
        self.weights[(neuron*(1+self._in*2) + nth) as usize]
    }

    #[allow(dead_code)]
    pub fn set_weight(&mut self, neuron: u32, nth: u32, value: f32) {
        self.weights[(neuron*(1+self._in*2) + nth) as usize] = value
    }

    pub fn process(&mut self, input: &[f32]) -> &Vec<f32> {
        // println!("processing {0} {1}", self.weights.len(), self.output.len());
        for i in 0..self._out {
            self.output[i as usize] = self.get_weight(i as u32, 0);
            for j in 0..self._in {
                let a = self.get_weight(i, 1+j);
                let b = self.get_weight(i, 1+self._in+j);
                self.output[i as usize]+= b / (1. + (input[j as usize] * a).exp());
            }
        }
        &self.output
    }
}

impl Mutable for Layer {

    fn mutate(&mut self) {
        if self.mrate > rand_float(0., 1.) {
            self.mrate+= rand_float(-0.01, 0.01);
            if self.mrate > 0.5 {
                self.mrate = 0.5
            } else if self.mrate < 0.00001 {
                self.mrate = 0.00001
            }
        }

        for i in 0..self.weights.len() {


            // if self.mrate > rand_float(0., 1.) {
            //     // let r = rand_float(-10., 10.);
            //     self.weights[i]+= rand_float(-1., 1.);
            //     self.weights[i]*= rand_float(0.5, 2.);
            // }

            if self.mrate > rand_float(0., 1.) {
                let r = rand_float(-1., 1.);
                self.weights[i]+= r*self.momento[i];
            }
            if self.mrate > rand_float(0., 1.) {
                self.momento[i]+= rand_float(-1., 1.);
            }


            // self.weights[i]*= rand_float(0.9, 1.11);
            // self.weights[i] = rand_float(-self.weights[i].abs(), self.weights[i].abs());
            // self.weights[i]+= self.momento[i]/2.;
            // self.momento[i]+= rand_float(-1., 1.)*0.1;
            // if rand_int(0, 20) == 0 {
            //     self.momento[i]*= rand_float(-2., 2.);
            // }
        }
    }

    fn crossover(&mut self, layer : &Self) {
        // for neuron in 0..self._out {
        //     let num = rand_int(0, 1);
        //     if num == 0 {
        //         for synapse in 0..1+self._in*2 {
        //             self.set_weight(neuron, synapse, layer.get_weight(neuron, synapse));
        //             // self.weights[i] = layer.weights[i];
        //         }
        //     }
        // }
        for i in 0..self.weights.len() {
            let num = rand_int(0, 1);
            if num == 0 {
                self.weights[i] = layer.weights[i];
                self.momento[i] = layer.momento[i];
            }
        }
    }
}




#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub definition: Vec<u32>,
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(definition : &[u32], mrate: f32) -> Network {
        let mut def: Vec<u32> = Vec::with_capacity(definition.len());
        def.extend(definition.iter());

        let mut layers : Vec<Layer> = Vec::new();
        for i in 1..definition.len() {
            layers.push(Layer::new(definition[i-1], definition[i], mrate));
        }
        Network { definition: def, layers }
    }

    #[allow(dead_code)]
    pub fn process(&mut self, input: &[f32]) -> &Vec<f32> {
        self.layers[0].process(input);
        for i in 1..self.layers.len() {
            let v = self.layers[i-1].output.clone();
            self.layers[i].process(&v);
        }
        let len = self.layers.len();
        &self.layers[len-1].output
    }

    #[allow(dead_code)]
    pub fn weight_count(&self) -> usize {
        let mut c = 0;
        for l in &self.layers {
            c+= l.weights.len()
        }
        c
    }

    #[allow(dead_code)]
    pub fn flat_weights(&self) -> Vec<f32> {

        let mut weights: Vec<f32> = Vec::new();

        for l in &self.layers {
            weights.extend(l.weights.iter());
        }

        weights
    }

    #[allow(dead_code)]
    pub fn input_len(&self) -> usize {
        self.layers[0]._in as usize
    }

    #[allow(dead_code)]
    pub fn output_len(&self) -> usize {
        self.layers[self.layers.len()-1]._out as usize
    }

    #[allow(dead_code)]
    pub fn set_mrate(&mut self, mrate: f32) {
        for l in &mut self.layers {
            l.mrate = mrate
        }
    }

    #[allow(dead_code)]
    pub fn fitness(&mut self, data: &[f32], labels: &[f32]) -> f32 {
        let mut fit = 0.;
        let _in = self.input_len();
        let _out = self.output_len();

        for i in 0..data.len()/_in {
            let data = &data[i*_in..(i+1)*_in];
            let labels = &labels[i*_out..(i+1)*_out];
            let res = self.process(data);
            for j in 0..res.len() {
                fit-= (res[j]-labels[j]).powf(2.)
            }
        }
        fit
    }
}


impl Mutable for Network {
    fn mutate(&mut self) {
        for l in &mut self.layers {
            l.mutate()
        }
    }
    fn crossover(&mut self, net: &Self) {
        for i in 0..self.layers.len() {
            self.layers[i].crossover(&net.layers[i])
        }
    }
}

pub fn ga<T: Mutable + Ord + Clone>(ancestors : &mut Vec<T>, new: &mut Vec<T>) {
    ancestors.sort_unstable();
    // new.push(ancestors[0].clone());
    while new.len() < ancestors.len() {
        let p = rand_int(0, (new.len() as f32).sqrt() as i32) as usize;
        let m = rand_int(0, (new.len() as f32).sqrt() as i32) as usize;
        let mut s = ancestors[p].clone();
        s.crossover(&mut ancestors[m]);
        s.mutate();
        new.push(s);
    }
}
