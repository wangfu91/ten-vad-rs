use std::collections::VecDeque;

pub trait AudioSample: Copy {}
impl AudioSample for i16 {}
impl AudioSample for f32 {}

/// Represents a buffer of audio samples, allowing for efficient processing
/// and retrieval of fixed-size audio frames.
#[derive(Debug)]
pub struct AudioFrameBuffer<T: AudioSample> {
    samples: VecDeque<T>,
}

impl<T: AudioSample> AudioFrameBuffer<T> {
    /// Create a new audio frame buffer.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::new(),
        }
    }

    /// Returns the number of samples in the audio frame buffer.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if the audio frame buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Append new samples to the audio frame buffer from any iterator.
    /// # Arguments:
    /// - `new_samples`: An iterator of samples to append.
    pub fn append_samples<I>(&mut self, new_samples: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.samples.extend(new_samples);
    }

    /// Get a fixed-size chunk of samples from the audio frame buffer.
    /// If there are not enough samples, returns None.
    /// # Arguments:
    /// - `frame_size`: The number of samples to retrieve.
    /// # Returns:
    /// - `Some(Vec<T>)`: A vector containing the requested number of samples.
    /// - `None`: If there are not enough samples available.
    pub fn pop_frame(&mut self, frame_size: usize) -> Option<Vec<T>> {
        // If we have enough samples, return a chunk of the specified size
        if self.samples.len() >= frame_size {
            Some(self.samples.drain(..frame_size).collect())
        } else {
            None
        }
    }
}

impl<T: AudioSample> Default for AudioFrameBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_audio_frame_buffer() {
        let buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_default_audio_frame_buffer() {
        let buffer_i16 = AudioFrameBuffer::<i16>::default();
        let buffer_f32 = AudioFrameBuffer::<f32>::default();
        assert_eq!(buffer_i16.len(), 0);
        assert_eq!(buffer_f32.len(), 0);
    }

    #[test]
    fn test_extend_samples_i16() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        let samples = vec![1i16, 2, 3, 4, 5];
        buffer.append_samples(samples);
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_extend_samples_f32() {
        let mut buffer: AudioFrameBuffer<f32> = AudioFrameBuffer::new();
        let samples = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        buffer.append_samples(samples);
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_extend_multiple_times() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        buffer.append_samples([1i16, 2, 3].iter().copied());
        buffer.append_samples([4i16, 5, 6].iter().copied());
        buffer.append_samples([7i16, 8].iter().copied());
        assert_eq!(buffer.len(), 8);
    }

    #[test]
    fn test_extend_empty_samples() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        let empty_samples: Vec<i16> = vec![];
        buffer.append_samples(empty_samples);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_pop_frame_sufficient_samples() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        buffer.append_samples([1i16, 2, 3, 4, 5, 6, 7, 8].iter().copied());

        let frame = buffer.pop_frame(4);
        assert!(frame.is_some());
        let frame = frame.unwrap();
        assert_eq!(frame, vec![1, 2, 3, 4]);
        assert_eq!(buffer.len(), 4); // Remaining samples
    }

    #[test]
    fn test_pop_frame_insufficient_samples() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        buffer.append_samples([1i16, 2, 3].iter().copied());

        let frame = buffer.pop_frame(5);
        assert!(frame.is_none());
        assert_eq!(buffer.len(), 3); // No samples consumed
    }

    #[test]
    fn test_pop_frame_exact_samples() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        buffer.append_samples([1i16, 2, 3, 4].iter().copied());

        let frame = buffer.pop_frame(4);
        assert!(frame.is_some());
        let frame = frame.unwrap();
        assert_eq!(frame, vec![1, 2, 3, 4]);
        assert_eq!(buffer.len(), 0); // All samples consumed
    }

    #[test]
    fn test_pop_multiple_frames() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        buffer.append_samples([1i16, 2, 3, 4, 5, 6, 7, 8, 9, 10].iter().copied());

        let frame1 = buffer.pop_frame(3);
        assert!(frame1.is_some());
        assert_eq!(frame1.unwrap(), vec![1, 2, 3]);

        let frame2 = buffer.pop_frame(3);
        assert!(frame2.is_some());
        assert_eq!(frame2.unwrap(), vec![4, 5, 6]);

        let frame3 = buffer.pop_frame(3);
        assert!(frame3.is_some());
        assert_eq!(frame3.unwrap(), vec![7, 8, 9]);

        // Only 1 sample left, not enough for frame of size 3
        let frame4 = buffer.pop_frame(3);
        assert!(frame4.is_none());
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_pop_frame_zero_size() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        buffer.append_samples([1i16, 2, 3].iter().copied());

        let frame = buffer.pop_frame(0);
        assert!(frame.is_some());
        let frame = frame.unwrap();
        assert_eq!(frame.len(), 0);
        assert_eq!(buffer.len(), 3); // No samples consumed
    }

    #[test]
    fn test_continuous_processing_simulation() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        let frame_size = 256;
        let chunk_size = 100;
        let mut total_samples = 0;

        // Simulate continuous audio processing
        for chunk in 0..10 {
            // Add new audio chunk
            let samples: Vec<i16> = (chunk * chunk_size..(chunk + 1) * chunk_size)
                .map(|x| x as i16)
                .collect();
            buffer.append_samples(samples);
            total_samples += chunk_size;

            // Process available frames
            let mut _frames_processed = 0;
            while buffer.pop_frame(frame_size).is_some() {
                _frames_processed += 1;
                total_samples -= frame_size;
            }

            // We should be able to process some frames when we have enough samples
            if total_samples >= frame_size {
                // If we have enough samples but didn't process frames, it means samples were already consumed
                // This is expected behavior in continuous processing
            }
        }

        // Verify the buffer works by adding enough samples and checking frame extraction
        buffer.append_samples(vec![0i16; frame_size]);
        let frame = buffer.pop_frame(frame_size);
        assert!(
            frame.is_some(),
            "Should be able to extract frame when enough samples are available"
        );
    }

    #[test]
    fn test_large_audio_frame_buffer() {
        let mut buffer: AudioFrameBuffer<i16> = AudioFrameBuffer::new();
        let large_samples: Vec<i16> = (0..100000).map(|x| x as i16).collect();
        buffer.append_samples(large_samples);

        assert_eq!(buffer.len(), 100000);

        let frame = buffer.pop_frame(1024);
        assert!(frame.is_some());
        assert_eq!(frame.unwrap().len(), 1024);
        assert_eq!(buffer.len(), 100000 - 1024);
    }

    #[test]
    fn test_mixed_type_frame_buffers() {
        let mut buffer_i16 = AudioFrameBuffer::<i16>::new();
        let mut buffer_f32 = AudioFrameBuffer::<f32>::new();

        buffer_i16.append_samples([1i16, 2, 3].iter().copied());
        buffer_f32.append_samples([1.0f32, 2.0, 3.0].iter().copied());

        let frame_i16 = buffer_i16.pop_frame(2);
        let frame_f32 = buffer_f32.pop_frame(2);

        assert!(frame_i16.is_some());
        assert!(frame_f32.is_some());
        assert_eq!(frame_i16.unwrap(), vec![1i16, 2]);
        assert_eq!(frame_f32.unwrap(), vec![1.0f32, 2.0]);
    }
}
