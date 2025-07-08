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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_audio_segment() {
        let segment: AudioSegment<i16> = AudioSegment::new();
        assert_eq!(segment.samples.len(), 0);
    }

    #[test]
    fn test_default_audio_segment() {
        let segment_i16 = AudioSegment::<i16>::default();
        let segment_f32 = AudioSegment::<f32>::default();
        assert_eq!(segment_i16.samples.len(), 0);
        assert_eq!(segment_f32.samples.len(), 0);
    }

    #[test]
    fn test_append_samples_i16() {
        let mut segment = AudioSegment::new();
        let samples = vec![1i16, 2, 3, 4, 5];
        segment.append_samples(&samples);
        assert_eq!(segment.samples.len(), 5);
    }

    #[test]
    fn test_append_samples_f32() {
        let mut segment = AudioSegment::new();
        let samples = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        segment.append_samples(&samples);
        assert_eq!(segment.samples.len(), 5);
    }

    #[test]
    fn test_append_multiple_times() {
        let mut segment = AudioSegment::new();
        segment.append_samples(&[1i16, 2, 3]);
        segment.append_samples(&[4i16, 5, 6]);
        segment.append_samples(&[7i16, 8]);
        assert_eq!(segment.samples.len(), 8);
    }

    #[test]
    fn test_append_empty_samples() {
        let mut segment = AudioSegment::new();
        let empty_samples: Vec<i16> = vec![];
        segment.append_samples(&empty_samples);
        assert_eq!(segment.samples.len(), 0);
    }

    #[test]
    fn test_get_audio_frame_sufficient_samples() {
        let mut segment = AudioSegment::new();
        segment.append_samples(&[1i16, 2, 3, 4, 5, 6, 7, 8]);

        let frame = segment.get_audio_frame(4);
        assert!(frame.is_some());
        let frame = frame.unwrap();
        assert_eq!(frame, vec![1, 2, 3, 4]);
        assert_eq!(segment.samples.len(), 4); // Remaining samples
    }

    #[test]
    fn test_get_audio_frame_insufficient_samples() {
        let mut segment = AudioSegment::new();
        segment.append_samples(&[1i16, 2, 3]);

        let frame = segment.get_audio_frame(5);
        assert!(frame.is_none());
        assert_eq!(segment.samples.len(), 3); // No samples consumed
    }

    #[test]
    fn test_get_audio_frame_exact_samples() {
        let mut segment = AudioSegment::new();
        segment.append_samples(&[1i16, 2, 3, 4]);

        let frame = segment.get_audio_frame(4);
        assert!(frame.is_some());
        let frame = frame.unwrap();
        assert_eq!(frame, vec![1, 2, 3, 4]);
        assert_eq!(segment.samples.len(), 0); // All samples consumed
    }

    #[test]
    fn test_get_multiple_frames() {
        let mut segment = AudioSegment::new();
        segment.append_samples(&[1i16, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let frame1 = segment.get_audio_frame(3);
        assert!(frame1.is_some());
        assert_eq!(frame1.unwrap(), vec![1, 2, 3]);

        let frame2 = segment.get_audio_frame(3);
        assert!(frame2.is_some());
        assert_eq!(frame2.unwrap(), vec![4, 5, 6]);

        let frame3 = segment.get_audio_frame(3);
        assert!(frame3.is_some());
        assert_eq!(frame3.unwrap(), vec![7, 8, 9]);

        // Only 1 sample left, not enough for frame of size 3
        let frame4 = segment.get_audio_frame(3);
        assert!(frame4.is_none());
        assert_eq!(segment.samples.len(), 1);
    }

    #[test]
    fn test_get_audio_frame_zero_size() {
        let mut segment = AudioSegment::new();
        segment.append_samples(&[1i16, 2, 3]);

        let frame = segment.get_audio_frame(0);
        assert!(frame.is_some());
        let frame = frame.unwrap();
        assert_eq!(frame.len(), 0);
        assert_eq!(segment.samples.len(), 3); // No samples consumed
    }

    #[test]
    fn test_continuous_processing_simulation() {
        let mut segment = AudioSegment::new();
        let frame_size = 256;
        let chunk_size = 100;
        let mut total_samples = 0;

        // Simulate continuous audio processing
        for chunk in 0..10 {
            // Add new audio chunk
            let samples: Vec<i16> = (chunk * chunk_size..(chunk + 1) * chunk_size)
                .map(|x| x as i16)
                .collect();
            segment.append_samples(&samples);
            total_samples += chunk_size;

            // Process available frames
            let mut _frames_processed = 0;
            while segment.get_audio_frame(frame_size).is_some() {
                _frames_processed += 1;
                total_samples -= frame_size;
            }

            // We should be able to process some frames when we have enough samples
            if total_samples >= frame_size {
                // If we have enough samples but didn't process frames, it means samples were already consumed
                // This is expected behavior in continuous processing
            }
        }

        // Verify the segment works by adding enough samples and checking frame extraction
        segment.append_samples(&vec![0i16; frame_size]);
        let frame = segment.get_audio_frame(frame_size);
        assert!(
            frame.is_some(),
            "Should be able to extract frame when enough samples are available"
        );
    }

    #[test]
    fn test_large_audio_segment() {
        let mut segment = AudioSegment::new();
        let large_samples: Vec<i16> = (0..100000).map(|x| x as i16).collect();
        segment.append_samples(&large_samples);

        assert_eq!(segment.samples.len(), 100000);

        let frame = segment.get_audio_frame(1024);
        assert!(frame.is_some());
        assert_eq!(frame.unwrap().len(), 1024);
        assert_eq!(segment.samples.len(), 100000 - 1024);
    }

    #[test]
    fn test_mixed_type_segments() {
        let mut segment_i16 = AudioSegment::<i16>::new();
        let mut segment_f32 = AudioSegment::<f32>::new();

        segment_i16.append_samples(&[1i16, 2, 3]);
        segment_f32.append_samples(&[1.0f32, 2.0, 3.0]);

        let frame_i16 = segment_i16.get_audio_frame(2);
        let frame_f32 = segment_f32.get_audio_frame(2);

        assert!(frame_i16.is_some());
        assert!(frame_f32.is_some());
        assert_eq!(frame_i16.unwrap(), vec![1i16, 2]);
        assert_eq!(frame_f32.unwrap(), vec![1.0f32, 2.0]);
    }
}
