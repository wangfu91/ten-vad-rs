use std::collections::VecDeque;

pub trait AudioSample: Copy {}
impl AudioSample for i16 {}
impl AudioSample for f32 {}

#[derive(Debug)]
pub struct AudioSegment<T: AudioSample> {
    samples: VecDeque<T>,
    target_size: usize,
}

impl<T: AudioSample> AudioSegment<T> {
    pub fn new(target_size: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(target_size * 2),
            target_size,
        }
    }

    pub fn append_samples(&mut self, new_samples: &[T]) -> Option<Vec<T>> {
        // Append new samples to buffer
        self.samples.extend(new_samples);

        // If we have enough samples, return a chunk
        if self.samples.len() >= self.target_size {
            Some(self.samples.drain(..self.target_size).collect())
        } else {
            None
        }
    }

    pub fn append_sample(&mut self, sample: T) -> Option<Vec<T>> {
        // Append single sample
        self.samples.push_back(sample);

        // If we have enough samples, return a chunk
        if self.samples.len() >= self.target_size {
            Some(self.samples.drain(..self.target_size).collect())
        } else {
            None
        }
    }
}
