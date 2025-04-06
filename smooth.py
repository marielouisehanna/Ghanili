import numpy as np
import soundfile as sf
import librosa
import librosa.effects
from scipy.signal import butter, filtfilt, lfilter
import scipy.ndimage
import warnings
warnings.filterwarnings('ignore')

class ArabicVocalProcessor:
    def __init__(self, input_file, output_file, sample_rate=44100):
        self.input_file = input_file
        self.output_file = output_file
        self.sample_rate = sample_rate
        print(f"Initializing processor for file: {input_file}")
        
    def process(self):
        """Main processing pipeline for Arabic vocal enhancement"""
        # Load the audio file
        print("Loading audio file...")
        vocals, sr = librosa.load(self.input_file, sr=self.sample_rate, mono=True)
        
        # Apply processing chain (without pitch correction and reverb)
        print("Cleaning and enhancing vocals...")
        processed = self.noise_reduction(vocals)
        processed = self.breath_removal(processed)
        # Pitch correction removed as requested
        processed = self.eq_enhancement(processed)
        processed = self.apply_compression(processed)
        # Reverb removed as requested
        
        # Final normalization
        processed = processed / (np.max(np.abs(processed)) + 1e-6) * 0.95
        
        # Save the processed audio
        print(f"Saving enhanced vocals to: {self.output_file}")
        sf.write(self.output_file, processed, self.sample_rate, subtype='PCM_24')
        print("Processing complete!")
        
        return self.output_file
    
    def noise_reduction(self, audio):
        """Advanced noise reduction preserving vocal characteristics"""
        print("- Applying noise reduction...")
        
        # Estimate noise profile from the quietest sections
        hop_length = 512
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        threshold = np.percentile(energy, 15)  # Use lowest 15% as noise
        
        # Create noise mask
        noise_mask = energy < threshold
        noise_indices = np.where(noise_mask)[0]
        
        if len(noise_indices) > 0:
            # Extract noise segments
            noise_segments = []
            for idx in noise_indices:
                start = idx * hop_length
                end = min(start + hop_length, len(audio))
                noise_segments.append(audio[start:end])
            
            # Calculate noise profile from these segments
            noise_profile = np.concatenate(noise_segments)
            if len(noise_profile) > 0:
                # Spectral subtraction
                n_fft = 2048
                audio_stft = librosa.stft(audio, n_fft=n_fft)
                noise_stft = librosa.stft(noise_profile, n_fft=n_fft)
                
                # Calculate average noise spectrum
                noise_spec = np.mean(np.abs(noise_stft)**2, axis=1)
                noise_spec = np.expand_dims(noise_spec, axis=1)
                
                # Subtract noise with oversubtraction factor
                oversubtraction = 2.0
                audio_spec = np.abs(audio_stft)**2
                clean_spec = np.maximum(audio_spec - oversubtraction * noise_spec, 0.0)
                
                # Apply phase of original signal
                audio_phase = np.angle(audio_stft)
                clean_stft = np.sqrt(clean_spec) * np.exp(1j * audio_phase)
                
                # Reconstruct time domain signal
                cleaned = librosa.istft(clean_stft, length=len(audio))
                return cleaned
        
        return audio
    
    def breath_removal(self, audio):
        """Remove breath sounds common in Arabic singing"""
        print("- Removing breath sounds...")
        
        hop_length = 256
        # Calculate spectral contrast for breath detection
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, hop_length=hop_length)
        # Breath detection using multiple features
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=hop_length)[0]
        
        # Breaths typically have lower energy, higher centroid, and lower contrast
        breath_likelihood = (1 - energy/np.max(energy)) * (spectral_centroid/np.max(spectral_centroid)) * (1 - np.mean(contrast[2:5], axis=0)/np.max(np.mean(contrast[2:5], axis=0)))
        breath_mask = breath_likelihood > np.percentile(breath_likelihood, 80)
        
        # Create attenuated mask for breaths
        attenuation_mask = np.ones(len(energy))
        attenuation_mask[breath_mask] = 0.3  # Reduce breath volume
        
        # Smooth transitions
        smooth_mask = scipy.ndimage.gaussian_filter1d(attenuation_mask, sigma=2)
        
        # Expand to audio length
        full_mask = np.repeat(smooth_mask, hop_length)[:len(audio)]
        if len(full_mask) < len(audio):
            full_mask = np.pad(full_mask, (0, len(audio) - len(full_mask)), mode='constant', constant_values=1)
        
        # Apply the mask to reduce breath sounds
        return audio * full_mask
    
    # Pitch correction method has been removed as requested
    # Keeping the method definition as a placeholder in case you want to add it back later
    def arabic_pitch_correction(self, audio):
        """Pitch correction tailored for Arabic maqam scales - DISABLED"""
        print("- Pitch correction disabled as requested")
        # Simply return the unmodified audio
        return audio
    
    def eq_enhancement(self, audio):
        """EQ tailored for Arabic vocals"""
        print("- Applying vocal EQ enhancement...")
        
        # Design filters for a 3-band parametric EQ
        def design_peaking_filter(center_freq, gain_db, q, fs):
            """Design a peaking filter"""
            w0 = 2 * np.pi * center_freq / fs
            alpha = np.sin(w0) / (2 * q)
            A = 10 ** (gain_db / 40)
            
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            
            return [b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0]
        
        # Apply a highpass filter to remove rumble
        def highpass(data, cutoff, fs, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return filtfilt(b, a, data)
        
        # Remove low-frequency rumble
        audio_hp = highpass(audio, cutoff=80, fs=self.sample_rate)
        
        # Enhance different frequency bands for Arabic vocals
        # 1. Add warmth (around 250Hz)
        b, a = design_peaking_filter(250, 2.0, 1.0, self.sample_rate)
        audio_eq1 = lfilter(b, a, audio_hp)
        
        # 2. Add presence (around 2kHz) - emphasizes Arabic vocal articulation
        b, a = design_peaking_filter(2000, 3.0, 1.0, self.sample_rate)
        audio_eq2 = lfilter(b, a, audio_eq1)
        
        # 3. Add air (around 8kHz)
        b, a = design_peaking_filter(8000, 1.5, 1.0, self.sample_rate)
        audio_eq3 = lfilter(b, a, audio_eq2)
        
        # 4. Reduce harshness (around 4-5kHz) which can be problematic in Arabic vocals
        b, a = design_peaking_filter(4500, -1.5, 1.5, self.sample_rate)
        audio_eq = lfilter(b, a, audio_eq3)
        
        return audio_eq
    
    def apply_compression(self, audio):
        """Apply compression with Arabic vocal characteristics in mind"""
        print("- Applying vocal compression...")
        
        # Compression parameters
        threshold = 0.3
        ratio = 4.0
        attack = 0.01  # seconds
        release = 0.2  # seconds
        makeup_gain = 1.5
        
        # Calculate attack and release in samples
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # Calculate gain reduction
        abs_audio = np.abs(audio)
        gain_reduction = np.ones_like(audio)
        
        for i in range(len(audio)):
            if abs_audio[i] > threshold:
                gain_reduction[i] = threshold + (abs_audio[i] - threshold) / ratio
                gain_reduction[i] = abs_audio[i] / gain_reduction[i]
            else:
                gain_reduction[i] = 1.0
        
        # Apply attack and release
        smoothed_gr = np.ones_like(gain_reduction)
        
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] < smoothed_gr[i-1]:  # Attack phase
                smoothed_gr[i] = (1 - 1/attack_samples) * smoothed_gr[i-1] + (1/attack_samples) * gain_reduction[i]
            else:  # Release phase
                smoothed_gr[i] = (1 - 1/release_samples) * smoothed_gr[i-1] + (1/release_samples) * gain_reduction[i]
        
        # Apply makeup gain
        compressed = audio * smoothed_gr * makeup_gain
        
        # Clip to prevent any potential overshoots
        compressed = np.clip(compressed, -0.98, 0.98)
        
        return compressed
    
    # Reverb method has been removed as requested
    # Keeping the method definition as a placeholder in case you want to add it back later
    def add_reverb(self, audio):
        """Add reverb suitable for Arabic vocals - DISABLED"""
        print("- Reverb effect disabled as requested")
        # Simply return the unmodified audio
        return audio


def process_arabic_vocals(input_file, output_file, sample_rate=44100):
    """Process Arabic vocals with all enhancements"""
    processor = ArabicVocalProcessor(input_file, output_file, sample_rate)
    return processor.process()


if __name__ == "__main__":
    # Example usage
    input_file = "processed_vocals2.wav"  # Replace with your input file
    output_file = "enhanced_arabic_vocals.wav"
    
    process_arabic_vocals(input_file, output_file)
