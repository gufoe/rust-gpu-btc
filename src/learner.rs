
use nn;
use std::cmp::Ordering;

#[derive(Clone, Serialize, Deserialize)]
pub struct Learner {
    pub fitness: f32,
    pub network: nn::Network,
}

impl nn::Mutable for Learner {
    fn mutate(&mut self) {
        self.network.mutate()
    }
    fn crossover(&mut self, l: &Self) {
        self.network.crossover(&l.network)
    }
}

impl PartialEq for Learner {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl Eq for Learner {}

impl PartialOrd for Learner {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Learner {
    fn cmp(&self, other: &Self) -> Ordering {
        if other.fitness > self.fitness {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}
