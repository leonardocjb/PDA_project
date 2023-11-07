
import pandas as pd
import numpy as np
import os
import glob
import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram

def preprocess():
    HOME_dir = "/cluster/projects/schwartzgroup/leo/PDA_project/"
    # Root directory where your speech data is located
    root_dir = HOME_dir + "data/Speech_data"

    # Lists to store paths of audio files for LAR and MIC contexts
    lar_audio_paths = []
    mic_audio_paths = []
    ref_f0_paths = []

    # Traverse through MALE and FEMALE folders
    for gender_folder in ["MALE", "FEMALE"]:
        gender_path = os.path.join(root_dir, gender_folder)

        # Traverse through LAR and MIC folders
        for context_folder in ["LAR", "MIC"]:
            context_path = os.path.join(gender_path, context_folder)

            # Traverse through individual speaker folders (F01 to F10 or M01 to M10)
            for speaker_folder in os.listdir(context_path):
                speaker_path = os.path.join(context_path, speaker_folder)

                # Use glob to find all WAV files in the speaker folder
                wav_files = glob.glob(os.path.join(speaker_path, "*.wav"))

                # Add the WAV file paths to the appropriate list
                if context_folder == "LAR":
                    lar_audio_paths.extend(wav_files)
                elif context_folder == "MIC":
                    mic_audio_paths.extend(wav_files)

        # Traverse through F01 to F10 or M01 to M10 subfolders within REF
        for ref_subfolder in os.listdir(os.path.join(gender_path, "REF")):
            ref_subfolder_path = os.path.join(gender_path, "REF", ref_subfolder)
            
            # Only process if the entry is a directory
            if os.path.isdir(ref_subfolder_path):
                f0_files = glob.glob(os.path.join(ref_subfolder_path, "*.f0"))
                ref_f0_paths.extend(f0_files)
        
    # Sort the lists of audio paths for each context
    lar_audio_paths = sorted(lar_audio_paths)
    mic_audio_paths = sorted(mic_audio_paths)
    ref_f0_paths = sorted(ref_f0_paths)
        


    # Combine all paths to a single list
    all_paths = []
    for i in range (len(lar_audio_paths)):
        all_paths.append([lar_audio_paths[i], mic_audio_paths[i], ref_f0_paths[i]])


    # Resampling factor
    resample_factor = 48000 // 12000  # 4

    # Window parameters
    window_size = int(0.032 * 12000)  # 32ms window size
    hop_size = int(0.010 * 12000)     # 10ms hop size

    # Load, resample, and convert audio files into overlapping windows
    resampler = Resample(orig_freq=48000, new_freq=12000)
    lar_windowed_audio = []
    mic_windowed_audio = []
    ref_freq = []

    for i in range (len(all_paths)):
        # waveform = resampler(torchaudio.load(all_paths[i][0])[0])
        # num_windows = (waveform.size(1) - window_size) // hop_size + 1
        # windows = [
        #     waveform[:, i * hop_size : i * hop_size + window_size] for i in range(num_windows)
        # ]
        # lar_windowed_audio.append(windows)

        
        waveform = resampler(torchaudio.load(all_paths[i][1])[0]).squeeze()
        num_windows = (waveform.size(0) - window_size) // hop_size + 1
        windows = torch.empty((num_windows, window_size), dtype=waveform.dtype)
        for j in range(num_windows):
            start = j * hop_size
            end = start + window_size
            windows[j, :] = waveform[start:end]
        
        mic_windowed_audio.append(windows)
        with open(all_paths[i][2], "r") as f:
            lines = f.readlines()
            first_column = [float(line.split()[0]) for line in lines]
        f.close()
        
        # pad extra 0 for each audio
        diff = len(windows) - len(first_column)
        first_column.extend([0] * diff)
        first_column = torch.FloatTensor(first_column).view(-1, 1)
        ref_freq.append(first_column)

    return mic_windowed_audio, ref_freq



