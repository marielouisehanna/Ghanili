import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
import os
import tempfile

class VocalAligner:
    """
    Simple vocal enhancer with manual time shifting
    """
    
    def __init__(self):
        self.sample_rate = 44100
        self.temp_dir = tempfile.mkdtemp()
    
    def __del__(self):
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def load_audio(self, file_path):
        """Load audio file and convert to mono"""
        audio_data, sample_rate = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio_data, sample_rate
    
    def shift_vocals(self, vocals, reference, shift_seconds):
        """
        Manually shift vocals by the specified number of seconds
        Positive values = shift vocals later (add silence at start)
        Negative values = shift vocals earlier (cut from beginning)
        """
        # Convert time shift to samples
        shift_samples = int(shift_seconds * self.sample_rate)
        
        # Create aligned output (same length as reference)
        aligned = np.zeros_like(reference)
        
        if shift_samples < 0:
            # Negative shift = cut beginning of vocals (shift earlier)
            cut_samples = abs(shift_samples)
            if cut_samples < len(vocals):
                copy_length = min(len(vocals) - cut_samples, len(aligned))
                aligned[:copy_length] = vocals[cut_samples:cut_samples + copy_length]
                print(f"Shifting vocals EARLIER by {abs(shift_seconds):.3f} seconds (cutting {cut_samples} samples from beginning)")
            else:
                print("Warning: Shift amount larger than vocal length!")
        else:
            # Positive shift = add silence at beginning (shift later)
            if shift_samples < len(aligned):
                copy_length = min(len(vocals), len(aligned) - shift_samples)
                aligned[shift_samples:shift_samples + copy_length] = vocals[:copy_length]
                print(f"Shifting vocals LATER by {shift_seconds:.3f} seconds (adding {shift_samples} samples of silence)")
            else:
                print("Warning: Shift amount larger than output length!")
        
        return aligned
    
    def enhance_voice(self, vocals):
        """Simple vocal enhancement"""
        # Make a copy to prevent modification of original
        enhanced = np.copy(vocals)
        
        # 1. De-essing (reduce harsh 's' sounds)
        nyquist = self.sample_rate / 2
        high_freq_low = 5000 / nyquist
        high_freq_high = 8000 / nyquist
        
        ess_filter = signal.butter(2, [high_freq_low, high_freq_high], btype='bandpass')
        ess_band = signal.filtfilt(ess_filter[0], ess_filter[1], enhanced)
        
        # Reduce sibilants by 30%
        enhanced = enhanced - (ess_band * 0.3)
        
        # 2. Add vocal presence (mid-range boost)
        presence_low = 1000 / nyquist
        presence_high = 3000 / nyquist
        
        presence_filter = signal.butter(2, [presence_low, presence_high], btype='bandpass')
        presence_band = signal.filtfilt(presence_filter[0], presence_filter[1], enhanced)
        
        # Add presence
        enhanced = enhanced + (presence_band * 0.15)
        
        # 3. Simple noise gate
        noise_floor = np.percentile(np.abs(enhanced), 5)
        gate = np.ones_like(enhanced)
        gate[np.abs(enhanced) < noise_floor] = 0.1
        
        # Apply gate
        enhanced = enhanced * gate
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced / max_val * 0.95
        
        return enhanced
    
    def process(self, vocals_file, reference_file, output_file=None, shift_seconds=0):
        """Main processing function with manual time shifting"""
        if output_file is None:
            output_file = os.path.join(self.temp_dir, "processed_vocals.wav")
        
        # Load audio files
        vocals, sr_vocals = self.load_audio(vocals_file)
        reference, sr_ref = self.load_audio(reference_file)
        
        print(f"Step 1: Applying manual time shift of {shift_seconds:.3f} seconds...")
        shifted_vocals = self.shift_vocals(vocals, reference, shift_seconds)
        
        print("Step 2: Enhancing voice quality...")
        enhanced_vocals = self.enhance_voice(shifted_vocals)
        
        print("Step 3: Matching levels...")
        # Match levels
        ref_level = np.sqrt(np.mean(reference**2) + 1e-10)
        vocals_level = np.sqrt(np.mean(enhanced_vocals**2) + 1e-10)
        
        gain = ref_level / vocals_level
        # Limit gain to reasonable range
        gain = max(0.5, min(2.0, gain))
        
        final_vocals = enhanced_vocals * gain
        
        # Save result
        sf.write(output_file, final_vocals, self.sample_rate, subtype='PCM_24')
        print(f"Processing complete! Enhanced vocals saved to: {output_file}")
        
        return output_file


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance vocals with manual time shifting")
    parser.add_argument("-v", "--vocals", required=True, help="Your vocals audio file")
    parser.add_argument("-r", "--reference", required=True, help="Reference track audio file")
    parser.add_argument("-o", "--output", help="Output audio file")
    parser.add_argument("-s", "--shift", type=float, default=0.0, 
                      help="Shift in seconds (negative = earlier, positive = later)")
    parser.add_argument("--character", action="store_true", help="Included for compatibility")
    
    args = parser.parse_args()
    
    aligner = VocalAligner()
    aligner.process(args.vocals, args.reference, args.output, args.shift)

# Usage examples:
# 
# To shift vocals 0.5 seconds earlier (cutting from beginning):
# python vocal_aligner.py -v albiML.wav -r albi.wav -o processed_vocals2.wav -s -0.3
#HAIDA EL SA7 EL DE8RE FAW2 HE 
# To shift vocals 0.3 seconds later (adding silence):
# python vocal_aligner.py -v your_vocals.wav -r reference.wav -o processed_vocals.wav -s 0.3