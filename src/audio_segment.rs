use std::collections::VecDeque;

pub trait AudioSample: Copy {}
impl AudioSample for i16 {}
impl AudioSample for f32 {}

/// Represents a segment of audio samples, allowing for efficient processing
/// and retrieval of fixed-size audio frames.
#[derive(Debug)]
pub struct AudioSegment<T: AudioSample> {
    samples: VecDeque<T>,
}

impl<T: AudioSample> AudioSegment<T> {
    /// Create a new audio segment.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::new(),
        }
    }

    /// Append new samples to the audio segment.
    /// # Arguments:
    /// - `new_samples`: A slice of samples to append.
    pub fn append_samples(&mut self, new_samples: &[T]) {
        self.samples.extend(new_samples);
    }

    /// Get a fixed-size chunk of samples from the audio segment.
    /// If there are not enough samples, returns None.
    /// # Arguments:
    /// - `frame_size`: The number of samples to retrieve.
    /// # Returns:
    /// - `Some(Vec<T>)`: A vector containing the requested number of samples.
    /// - `None`: If there are not enough samples available.
    pub fn get_audio_frame(&mut self, frame_size: usize) -> Option<Vec<T>> {
        // If we have enough samples, return a chunk of the specified size
        if self.samples.len() >= frame_size {
            Some(self.samples.drain(..frame_size).collect())
        } else {
            None
        }
    }
}

impl Default for AudioSegment<i16> {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AudioSegment<f32> {
    fn default() -> Self {
        Self::new()
    }
}
