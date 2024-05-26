use std::time::Instant;
use std::sync::OnceLock;
use std::io::{Read, Seek, BufReader, Result, Error};
use std::fs::File;
use owo_colors::OwoColorize;
use nanorand::{WyRand, Rng};

const LABELS_MAGIC: u32 = 2049;
const IMAGES_MAGIC: u32 = 2051;
const IMAGE_WIDTH: u32 = 28;
const IMAGE_SIZE: u32 = IMAGE_WIDTH * IMAGE_WIDTH;
const TRAIN_STEPS: u32 = 100;

type Image = [f32; IMAGE_SIZE as _];

fn main() {
  START.set(Instant::now()).unwrap();
  if let Err(e) = run() {
    log!("{}", e.red());
  }
}

fn run() -> Result<()> {
  let train_dataset = Dataset::load(
    File::open("train-labels.idx1-ubyte")?,
    File::open("train-images.idx3-ubyte")?,
  )?;
  log!("loaded {} training images", train_dataset.len());

  let mut network = Network::rand();
  log!("training with {} steps", TRAIN_STEPS);
  for i in 0..TRAIN_STEPS {
    let (loss, acc) = network.train_step(&train_dataset);
    log!("iter {} loss={:.2} acc={:.2}%", i + 1, loss, acc * 100.0);
  }

  let test_dataset = Dataset::load(
    File::open("t10k-labels.idx1-ubyte")?,
    File::open("t10k-images.idx3-ubyte")?,
  )?;
  log!("loaded {} test images", test_dataset.len());
  let acc = network.test_accuracy(&test_dataset);
  log!("accuracy on test dataset {:.2}%", acc * 100.0);

  Ok(())
}

struct Network {
  biases: [f32; 10],
  weights: [[f32; IMAGE_SIZE as _]; 10],
}

impl Network {
  fn zero() -> Self {
    Self {
      biases: [0.0; 10],
      weights: [[0.0; IMAGE_SIZE as _]; 10],
    }
  }

  fn rand() -> Self {
    let mut rng = WyRand::new();
    Self {
      biases: [0.0; 10].map(|_| rng.generate()),
      weights: [[0.0; IMAGE_SIZE as _]; 10].map(|i| i.map(|_| rng.generate())),
    }
  }

  fn train_step(&mut self, dataset: &Dataset) -> (f32, f32) {
    let mut gradient = Network::zero();
    let rate = 0.5;

    let mut loss = 0.0;
    let mut correct = 0.0;
    for (label, img) in &dataset.0 {
      let activations = self.hypothesis(img);
      for b in 0..10 {
        let mut p = activations[b];
        if b == *label as _ {
          p -= 1.0;
        }
        gradient.biases[b] += p;
        for w in 0..IMAGE_SIZE as _ {
          gradient.weights[b][w] += p * img[w];
        }
      }
      if max_idx(activations) == *label {
        correct += 1.0;
      }
      loss -= activations[*label as usize].ln();
    }

    for b in 0..10 {
      self.biases[b] -= rate * gradient.biases[b] / dataset.len() as f32;
      for w in 0..IMAGE_SIZE as _ {
        self.weights[b][w] -= rate * gradient.weights[b][w] / dataset.len() as f32;
      }
    }

    (loss / dataset.len() as f32, correct / dataset.len() as f32)
  }

  fn test_accuracy(&self, dataset: &Dataset) -> f32 {
    let mut correct = 0;
    for (c, i) in &dataset.0 {
      if *c == max_idx(self.hypothesis(i)) {
        correct += 1;
      }
    }
    correct as f32 / dataset.len() as f32
  }

  fn hypothesis(&self, img: &Image) -> [f32; 10] {
    let mut out = self.biases;
    for b in 0..10 {
      out[b] += self.weights[b]
        .iter()
        .enumerate()
        .map(|(j, w)| w * img[j])
        .sum::<f32>();
    }
    softmax(out)
  }
}

fn max_idx(a: [f32; 10]) -> u8 {
  a.iter()
    .enumerate()
    .max_by(|x, y| x.1.total_cmp(y.1))
    .unwrap()
    .0 as u8
}

fn softmax(a: [f32; 10]) -> [f32; 10] {
  let max = a.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
  let sum = a.iter().map(|x| (x - max).exp()).sum::<f32>();
  a.map(|x| (x - max).exp() / sum)
}

struct Dataset(Vec<(u8, Image)>);

impl Dataset {
  fn load<R: Read + Seek>(labels: R, images: R) -> Result<Self> {
    let mut labels = BufReader::new(labels);
    let mut images = BufReader::new(images);

    if read_u32(&mut labels)? != LABELS_MAGIC || read_u32(&mut images)? != IMAGES_MAGIC {
      return Err(Error::other("invalid magic"));
    }
    let len = read_u32(&mut labels)?;
    if len != read_u32(&mut images)? {
      return Err(Error::other("mismatched lengths"));
    }
    if read_u32(&mut images)? != IMAGE_WIDTH || read_u32(&mut images)? != IMAGE_WIDTH {
      return Err(Error::other("unexpected dimensions"));
    }

    let mut dataset = Vec::with_capacity(len as _);
    for _ in 0..len {
      let label = read_u8(&mut labels)?;
      let mut buf = [0; IMAGE_SIZE as _];
      images.read_exact(&mut buf)?;
      let image = buf.map(|px| px as f32 / u8::MAX as f32);
      dataset.push((label, image));
    }

    Ok(Self(dataset))
  }

  fn len(&self) -> usize {
    self.0.len()
  }
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
  let mut buf = [0; 4];
  reader.read_exact(&mut buf)?;
  Ok(u32::from_be_bytes(buf))
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
  let mut buf = [0; 1];
  reader.read_exact(&mut buf)?;
  Ok(u8::from_be_bytes(buf))
}

static START: OnceLock<Instant> = OnceLock::new();

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        println!("{} {}", format!("[{: >12.6}]", START.get().unwrap().elapsed().as_secs_f32()).green(), format!($($arg)*));
    };
}
