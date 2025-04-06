import numpy as np
import librosa
import soundfile as sf
import argparse
from scipy import signal
import matplotlib.pyplot as plt
import os


class VocalMixer:
    """A class to mix enhanced vocals with instrumental accompaniment."""
    
    def __init__(self, sample_rate=44100, use_output_dir=False):
        """
        Initialize the VocalMixer.
        
        Args:
            sample_rate: Sample rate of audio processing
            use_output_dir: Whether to use the output directory or current directory
        """
        self.sr = sample_rate
        self.use_output_dir = use_output_dir
        
        # Create output directory if needed and if it doesn't exist
        if self.use_output_dir:
            os.makedirs("output", exist_ok=True)
    
    def load_audio_files(self, vocals_path, accompaniment_path):
        """
        Load the vocal and accompaniment audio files with full precision.
        
        Args:
            vocals_path: Path to the vocal audio file
            accompaniment_path: Path to the accompaniment audio file
            
        Returns:
            vocals: Vocal audio as numpy array
            accompaniment: Accompaniment audio as numpy array
        """
        print(f"Loading vocals from '{os.path.basename(vocals_path)}'")
        vocals, sr_vocals = librosa.load(vocals_path, sr=self.sr, duration=None)
        
        print(f"Loading accompaniment from '{os.path.basename(accompaniment_path)}'")
        accompaniment, sr_accomp = librosa.load(accompaniment_path, sr=self.sr, duration=None)
        
        # Report exact track durations
        vocal_duration = len(vocals) / self.sr
        accomp_duration = len(accompaniment) / self.sr
        print(f"Vocals duration: {vocal_duration:.6f} seconds ({len(vocals)} samples)")
        print(f"Accompaniment duration: {accomp_duration:.6f} seconds ({len(accompaniment)} samples)")
        
        return vocals, accompaniment
    
    def align_lengths(self, vocals, accompaniment):
        """
        Ensure both tracks are exactly the same length without trimming by padding the shorter track.
        This preserves the full duration of both tracks.
        
        Args:
            vocals: Vocal audio as numpy array
            accompaniment: Accompaniment audio as numpy array
            
        Returns:
            vocals: Padded vocals if needed
            accompaniment: Padded accompaniment if needed
        """
        print("Making tracks exactly the same length by padding the shorter one...")
        
        # Get lengths in samples
        vocal_length = len(vocals)
        accompaniment_length = len(accompaniment)
        
        # Calculate length difference in seconds
        length_diff_sec = abs(vocal_length - accompaniment_length) / self.sr
        print(f"Length difference between tracks: {length_diff_sec:.2f} seconds")
        
        # Determine the longer track and pad the shorter one to match
        if vocal_length < accompaniment_length:
            # Vocals are shorter, pad with silence
            print(f"Padding vocals with {length_diff_sec:.2f} seconds of silence")
            padding = np.zeros(accompaniment_length - vocal_length)
            vocals = np.concatenate([vocals, padding])
        elif accompaniment_length < vocal_length:
            # Accompaniment is shorter, pad with silence
            print(f"Padding accompaniment with {length_diff_sec:.2f} seconds of silence")
            padding = np.zeros(vocal_length - accompaniment_length)
            accompaniment = np.concatenate([accompaniment, padding])
        else:
            print("Tracks are already the same length - no adjustment needed")
        
        # Double-check that lengths match exactly now
        assert len(vocals) == len(accompaniment), "Track lengths still don't match!"
        print(f"Both tracks now exactly {len(vocals) / self.sr:.2f} seconds long")
        
        return vocals, accompaniment
    
    def apply_ducking(self, vocals, accompaniment, threshold=0.05, reduction=0.3, attack=0.02, release=0.3):
        """
        Apply much gentler dynamic ducking with higher threshold and lower reduction.
        
        Args:
            vocals: Vocal audio as numpy array
            accompaniment: Accompaniment audio as numpy array
            threshold: Volume threshold to trigger ducking (HIGHER = less sensitive)
            reduction: Amount to reduce volume (LOWER = less reduction)
            attack: Attack time in seconds (HIGHER = more gradual)
            release: Release time in seconds (HIGHER = more gradual)
            
        Returns:
            ducked_accompaniment: Ducking-adjusted accompaniment
        """
        print("Applying very gentle ducking (minimal accompaniment changes)...")
        
        # Create a control signal based on vocal volume
        vocal_envelope = np.abs(vocals)
        
        # Smooth the envelope
        b, a = signal.butter(2, 10 / (self.sr / 2), 'lowpass')
        vocal_envelope = signal.filtfilt(b, a, vocal_envelope)
        
        # Create ducking envelope - starting with no reduction
        ducking_mask = np.ones_like(vocal_envelope)
        
        # Where vocals are SIGNIFICANTLY above threshold, apply MINIMAL reduction
        active_regions = vocal_envelope > threshold
        ducking_mask[active_regions] = 1.0 - reduction
        
        # Apply LONGER attack and release smoothing for very gradual changes
        attack_samples = int(attack * self.sr)  # Longer attack
        release_samples = int(release * self.sr)  # Longer release
        
        if attack_samples > 0 or release_samples > 0:
            # Smooth the transitions with a low-pass filter
            b, a = signal.butter(1, 1 / (max(attack_samples, release_samples) * 2), 'lowpass')
            ducking_mask = signal.filtfilt(b, a, ducking_mask)
        
        # Apply the very gentle ducking mask to the accompaniment
        ducked_accompaniment = accompaniment * ducking_mask
        
        return ducked_accompaniment
    
    def mix_tracks(self, vocals, accompaniment, vocal_gain=1.5, accompaniment_gain=0.6):
        """
        Mix the vocal and accompaniment tracks with enhanced integration and phase alignment.
        
        Args:
            vocals: Vocal audio as numpy array
            accompaniment: Accompaniment audio as numpy array
            vocal_gain: Gain factor for vocals (0-infinity, higher = louder vocals)
            accompaniment_gain: Gain factor for accompaniment (0-infinity, lower = quieter background)
            
        Returns:
            mixed: Mixed audio with perfect integration
        """
        print(f"Mixing with enhanced vocal integration (gain: {vocal_gain:.2f}, accompaniment gain: {accompaniment_gain:.2f})")
        
        # Apply stereo enhancement to create space for vocals
        # First convert mono to stereo if needed (assuming accompaniment might be stereo)
        if accompaniment.ndim == 1:
            # Create a basic stereo spread for mono accompaniment
            print("Converting accompaniment to stereo for better spatial integration")
            left = accompaniment * 0.95  # Slightly quieter on left
            right = accompaniment * 1.05  # Slightly louder on right
            accompaniment_stereo = np.vstack((left, right))
            
            # Apply a subtle delay to one channel for width
            delay_samples = int(0.01 * self.sr)  # 10ms delay
            right = np.pad(right, (delay_samples, 0))[:-delay_samples]
            accompaniment_stereo = np.vstack((left, right))
            
            # Convert back to mono for mixing with vocals
            accompaniment = np.mean(accompaniment_stereo, axis=0)
        
        # Apply a subtle high-pass filter to vocals to remove low-end mud
        sos = signal.butter(2, 100, 'highpass', fs=self.sr, output='sos')
        vocals_filtered = signal.sosfilt(sos, vocals)
        
        # Apply a subtle resonance to vocals for better intelligibility
        vocal_mid_boost = signal.butter(2, [800, 3000], 'bandpass', fs=self.sr, output='sos')
        vocals_res = signal.sosfilt(vocal_mid_boost, vocals_filtered)
        vocals_enhanced = 0.7 * vocals_filtered + 0.3 * vocals_res
        
        # Apply gain
        vocals_adjusted = vocals_enhanced * vocal_gain
        accompaniment_adjusted = accompaniment * accompaniment_gain
        
        # Phase alignment - ensure vocals don't cancel out important frequencies
        # Apply a subtle phase alignment technique
        vocals_phase = np.fft.rfft(vocals_adjusted)
        accompaniment_phase = np.fft.rfft(accompaniment_adjusted)
        
        # Calculate phase difference
        vocals_phase_angle = np.angle(vocals_phase)
        accompaniment_phase_angle = np.angle(accompaniment_phase)
        phase_diff = vocals_phase_angle - accompaniment_phase_angle
        
        # Apply subtle phase correction to critical vocal frequencies (800Hz-3kHz)
        freq_bins = np.fft.rfftfreq(len(vocals_adjusted), 1/self.sr)
        vocal_range_mask = (freq_bins >= 800) & (freq_bins <= 3000)
        phase_correction = np.ones_like(vocals_phase, dtype=complex)
        
        # For frequencies in critical range where phases are very close (could cause cancelation)
        close_phase_mask = vocal_range_mask & (np.abs(phase_diff) < 0.1)
        phase_correction[close_phase_mask] = np.exp(1j * 0.2)  # Slight phase shift
        
        # Apply the phase correction
        vocals_phase_fixed = vocals_phase * phase_correction
        vocals_adjusted = np.fft.irfft(vocals_phase_fixed, len(vocals_adjusted))
        
        # Mix tracks with improved integration
        mixed = vocals_adjusted + accompaniment_adjusted
        
        # Prevent clipping with smoother limiting
        def soft_clip(signal, threshold=0.9):
            """Soft clipper for more transparent limiting"""
            result = signal.copy()
            mask = np.abs(signal) > threshold
            # Apply a smooth curve instead of hard clipping
            result[mask] = np.sign(signal[mask]) * (
                threshold + (1 - threshold) * np.tanh((np.abs(signal[mask]) - threshold) / (1 - threshold))
            )
            return result
        
        max_value = np.max(np.abs(mixed))
        if max_value > 0.95:
            print(f"Applying transparent limiting (peak level: {max_value:.2f})")
            mixed = soft_clip(mixed)
            # Ensure we don't exceed 0dB
            mixed = mixed / np.max(np.abs(mixed)) * 0.95
        
        return mixed
    
    def apply_eq_and_mastering(self, mixed_audio):
        """
        Apply final EQ and mastering to the mixed track with vocal enhancement.
        
        Args:
            mixed_audio: Mixed audio as numpy array
            
        Returns:
            mastered: Mastered audio
        """
        print("Applying vocal-focused EQ and mastering...")
        
        # Apply multi-band EQ
        stft = librosa.stft(mixed_audio)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Sub bass (< 80 Hz)
        sub_mask = freqs < 80
        stft[sub_mask] *= 0.95  # Slight reduction to emphasize vocals
        
        # Bass (80-250 Hz)
        bass_mask = (freqs >= 80) & (freqs < 250)
        stft[bass_mask] *= 0.9  # More reduction in bass range
        
        # Low mids (250-500 Hz)
        low_mid_mask = (freqs >= 250) & (freqs < 500)
        stft[low_mid_mask] *= 0.9  # Cut to avoid muddiness
        
        # Mids (500-2000 Hz) - Vocal fundamental range
        mid_mask = (freqs >= 500) & (freqs < 2000)
        stft[mid_mask] *= 1.2  # Stronger boost for vocal presence
        
        # High mids (2000-5000 Hz) - Vocal clarity range
        high_mid_mask = (freqs >= 2000) & (freqs < 5000)
        stft[high_mid_mask] *= 1.25  # Enhanced boost for vocal clarity
        
        # Highs (5000-12000 Hz)
        high_mask = (freqs >= 5000) & (freqs < 12000)
        stft[high_mask] *= 1.1  # Boost for air and presence
        
        # Very high (> 12000 Hz)
        very_high_mask = freqs >= 12000
        stft[very_high_mask] *= 1.0  # Keep as is
        
        # Convert back to time domain
        eq_audio = librosa.istft(stft)
        
        # Apply mild compression with focus on preserving vocal dynamics
        def compress_audio(audio, threshold=0.25, ratio=2.5):
            """Compressor function tuned for vocal clarity"""
            compressed = np.copy(audio)
            mask = np.abs(audio) > threshold
            compressed[mask] = np.sign(audio[mask]) * (
                threshold + (np.abs(audio[mask]) - threshold) / ratio
            )
            return compressed
        
        compressed = compress_audio(eq_audio)
        
        # Apply limiting to maximize loudness
        def limit_audio(audio, threshold=0.95):
            """Limiter function with higher threshold for vocal clarity"""
            limited = np.copy(audio)
            scale = min(threshold / (np.max(np.abs(audio)) + 1e-10), 1.0)
            limited = audio * scale
            return limited
        
        limited = limit_audio(compressed)
        
        # Normalize to industry standard loudness
        mastered = limited / np.max(np.abs(limited)) * 0.95
        
        return mastered
    
    def plot_waveforms(self, vocals, accompaniment, mixed_output, output_path):
        """
        Generate a plot comparing the input and output waveforms.
        
        Args:
            vocals: Vocal audio as numpy array
            accompaniment: Accompaniment audio as numpy array
            mixed_output: Final mixed output as numpy array
            output_path: Base path for the visualization
        """
        print("Generating waveform visualization...")
        
        # Make sure all arrays have the same length for plotting
        min_length = min(len(vocals), len(accompaniment), len(mixed_output))
        vocals_plot = vocals[:min_length]
        accompaniment_plot = accompaniment[:min_length]
        mixed_output_plot = mixed_output[:min_length]
        
        # Calculate time axis for plotting
        time = np.linspace(0, min_length / self.sr, min_length)
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Plot vocals
        axs[0].plot(time, vocals_plot, color='blue', alpha=0.7)
        axs[0].set_title('Vocals')
        axs[0].set_ylabel('Amplitude')
        
        # Plot accompaniment
        axs[1].plot(time, accompaniment_plot, color='green', alpha=0.7)
        axs[1].set_title('Accompaniment')
        axs[1].set_ylabel('Amplitude')
        
        # Plot mixed output
        axs[2].plot(time, mixed_output_plot, color='red', alpha=0.7)
        axs[2].set_title('Mixed Output')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Amplitude')
        
        # Tight layout
        plt.tight_layout()
        
        # Determine visualization save path
        base_name = os.path.splitext(output_path)[0]
        viz_path = f"{base_name}_waveform.png"
        
        # Save the plot
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Waveform visualization saved to '{viz_path}'")
    
    def mix(self, vocals_path, accompaniment_path, output_path, 
            vocal_gain=1.5, accompaniment_gain=0.6, apply_ducking=True):
        """
        Main function to mix vocals with accompaniment with guaranteed full duration.
        
        Args:
            vocals_path: Path to vocal audio file
            accompaniment_path: Path to accompaniment audio file
            output_path: Path for saving the mixed output
            vocal_gain: Gain factor for vocals (0-infinity)
            accompaniment_gain: Gain factor for accompaniment (0-infinity)
            apply_ducking: Whether to apply ducking
            
        Returns:
            output_path: Path to the processed file
        """
        print("\n--- Starting Vocal-Focused Audio Mixing Process ---\n")
        
        # Load audio files
        vocals, accompaniment = self.load_audio_files(vocals_path, accompaniment_path)
        
        # Store original lengths for verification
        orig_vocal_len = len(vocals)
        orig_accomp_len = len(accompaniment)
        orig_vocal_duration = orig_vocal_len / self.sr
        orig_accomp_duration = orig_accomp_len / self.sr
        
        # Determine expected output length (should be the max of both inputs)
        expected_duration = max(orig_vocal_duration, orig_accomp_duration)
        expected_samples = int(expected_duration * self.sr)
        
        # Make sure both tracks have exactly the same length
        vocals, accompaniment = self.align_lengths(vocals, accompaniment)
        
        # Verify lengths match expected output length
        if len(vocals) < expected_samples:
            # Pad both tracks to expected length if there's any rounding error
            padding = expected_samples - len(vocals)
            if padding > 0:
                print(f"Adding {padding} samples padding to ensure full duration")
                vocals = np.pad(vocals, (0, padding))
                accompaniment = np.pad(accompaniment, (0, padding))
        
        # Apply ducking if requested
        if apply_ducking:
            accompaniment = self.apply_ducking(vocals, accompaniment, threshold=0.05, reduction=0.3)
        
        # Mix tracks with vocal emphasis
        mixed = self.mix_tracks(vocals, accompaniment, vocal_gain, accompaniment_gain)
        
        # Apply final EQ and mastering with vocal focus
        final_mix = self.apply_eq_and_mastering(mixed)
        
        # Verify the final output length (should be exactly the expected length)
        final_duration = len(final_mix) / self.sr
        print(f"Final mix duration: {final_duration:.6f} seconds ({len(final_mix)} samples)")
        print(f"Original vocal duration: {orig_vocal_duration:.6f} seconds")
        print(f"Original accompaniment duration: {orig_accomp_duration:.6f} seconds")
        
        if abs(final_duration - expected_duration) > 0.01:
            print(f"WARNING: Final duration {final_duration:.2f}s differs from expected {expected_duration:.2f}s")
            print("Forcing output to exact expected length...")
            
            # Force the exact length by padding or trimming to the precise expected size
            if len(final_mix) < expected_samples:
                # Pad to expected length
                final_mix = np.pad(final_mix, (0, expected_samples - len(final_mix)))
            elif len(final_mix) > expected_samples:
                # Trim to expected length
                final_mix = final_mix[:expected_samples]
                
            print(f"Adjusted final mix duration: {len(final_mix) / self.sr:.6f} seconds ({len(final_mix)} samples)")
        
        # Save output
        print(f"Saving final mix to '{output_path}'")
        sf.write(output_path, final_mix, self.sr)
        
        try:
            # Generate visualization - if this fails, continue with the rest of the process
            self.plot_waveforms(vocals, accompaniment, final_mix, output_path)
        except Exception as e:
            print(f"Warning: Could not generate waveform visualization: {e}")
            print("Continuing with the mixing process...")
        
        print("\n--- Vocal-Focused Mixing Complete! ---\n")
        print(f"Output file duration: {len(final_mix) / self.sr:.2f} seconds")
        return output_path


def main():
    """Main function to handle CLI arguments and process files."""
    parser = argparse.ArgumentParser(description="Mix vocals with accompaniment with vocal emphasis")
    
    parser.add_argument("--vocals", "-v", required=True, 
                        help="Path to vocal audio file")
    parser.add_argument("--accompaniment", "-a", required=True,
                        help="Path to accompaniment audio file")
    parser.add_argument("--output", "-o",
                        help="Path for output file (defaults to mixed_[vocal_filename])")
    parser.add_argument("--vocal-gain", "-vg", type=float, default=1.5,
                        help="Gain for vocals (0-infinity)")
    parser.add_argument("--accompaniment-gain", "-ag", type=float, default=0.6,
                        help="Gain for accompaniment (0-infinity)")
    parser.add_argument("--no-ducking", dest="ducking", action="store_false",
                        help="Disable ducking effect")
    parser.add_argument("--no-visualization", dest="visualization", action="store_false",
                        help="Disable waveform visualization")
    parser.add_argument("--sample-rate", "-sr", type=int, default=44100,
                        help="Sample rate for processing")
    parser.add_argument("--use-output-dir", action="store_true",
                        help="Save output to 'output' directory instead of current directory")
    
    parser.set_defaults(ducking=True, visualization=True, use_output_dir=False)
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        vocal_basename = os.path.basename(args.vocals)
        vocal_filename = os.path.splitext(vocal_basename)[0]
        accomp_basename = os.path.basename(args.accompaniment)
        accomp_filename = os.path.splitext(accomp_basename)[0]
        
        if args.use_output_dir:
            output_path = f"output/{vocal_filename}_mixed_with_{accomp_filename}.wav"
        else:
            output_path = f"{vocal_filename}_mixed_with_{accomp_filename}.wav"
    
    # Create VocalMixer instance
    mixer = VocalMixer(sample_rate=args.sample_rate, use_output_dir=args.use_output_dir)
    
    # Mix the tracks
    mixer.mix(
        args.vocals,
        args.accompaniment,
        output_path,
        args.vocal_gain,
        args.accompaniment_gain,
        args.ducking
    )


if __name__ == "__main__":
    # You can either use command line arguments:
    # main()
    
    # Or specify parameters directly in the code:
    # Simply modify these lines to run without command-line arguments
    
    # Direct usage example:
    mixer = VocalMixer(sample_rate=44100)
    
    # Specify your input and output files here
    vocals_file = "output.wav"  # Replace with your vocals file
    accompaniment_file = "accompaniment.wav"  # Replace with your accompaniment file
    output_file = "final.wav"  # Specify your desired output filename
    
    # Modified parameters for subtle accompaniment changes
    # Higher vocal gain but MUCH gentler ducking
    vocal_gain = 1.7  # Slightly stronger vocals
    accompaniment_gain = 0.8  # Less reduction on accompaniment
    apply_ducking = True  # Much gentler ducking is now applied
    
    # Process the mix
    mixer.mix(
        vocals_file,
        accompaniment_file,
        output_file,
        vocal_gain,
        accompaniment_gain,
        apply_ducking
    )