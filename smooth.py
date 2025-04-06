import numpy as np
import soundfile as sf
import librosa

def smooth_noise_gate(vocals, sample_rate=44100, threshold_percentile=20, singing_start_threshold=0.8):
    hop = 512
    energy = librosa.feature.rms(y=vocals, hop_length=hop)[0]
    threshold = np.percentile(energy, threshold_percentile)

    # Normalize energy to get a soft mask for quieter parts
    norm_energy = np.clip((energy - threshold) / (np.max(energy) - threshold + 1e-6), 0, 1)
    smoothed_mask = np.repeat(norm_energy, hop)[:len(vocals)]
    if len(smoothed_mask) < len(vocals):
        smoothed_mask = np.pad(smoothed_mask, (0, len(vocals) - len(smoothed_mask)), mode='constant')

    # Detect when singing begins based on energy threshold
    singing_frame = None
    for i, val in enumerate(norm_energy):
        if val >= singing_start_threshold:
            singing_frame = i
            break

    # After singing starts, blend the mask with full bypass (1.0)
    if singing_frame is not None:
        start_index = singing_frame * hop
        smoothed_mask[start_index:] = (smoothed_mask[start_index:] + 1.0) / 2

    return vocals * smoothed_mask

def process_audio(input_file, output_file, sample_rate=44100):
    vocals, sr = librosa.load(input_file, sr=sample_rate, mono=True)
    cleaned = smooth_noise_gate(vocals, sr)
    cleaned /= (np.max(np.abs(cleaned)) + 1e-6) * 0.98
    sf.write(output_file, cleaned, sample_rate, subtype='PCM_24')
    print(f"Cleaned vocals saved to: {output_file}")

# Example usage:
# process_audio('input.wav', 'output.wav')

process_audio('processed_vocals2.wav', 'output.wav')
