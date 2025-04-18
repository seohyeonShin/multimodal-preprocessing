#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import librosa
import pyworld as pw
from typing import Dict, Any, Tuple, Optional, List
from tqdm import tqdm

from core.base import Processor, processor_registry
from core.utils import time_it, get_device

# Global variables for mel processing
mel_basis = {}
hann_window = {}

def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Convert waveform to mel spectrogram using PyTorch
    
    Args:
        y: Audio waveform [B, T]
        n_fft: FFT size
        num_mels: Number of mel bins
        sampling_rate: Audio sampling rate
        hop_size: Hop size
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to pad signal at the beginning and end
        
    Returns:
        mel_spectrogram: Mel spectrogram [B, n_mels, T']
        energy: Energy [B, T']
    """
    global mel_basis, hann_window
    
    device = y.device
    
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
        
    # Initialize mel basis and window if not already done
    if fmax not in mel_basis:
        from librosa.filters import mel as librosa_mel_fn
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(device)] = torch.from_numpy(mel).float().to(device)
        hann_window[str(device)] = torch.hann_window(win_size).to(device)
    
    # Padding for STFT
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect'
    ).squeeze(1)
    
    # STFT
    spec = torch.stft(
        y, n_fft, 
        hop_length=hop_size, 
        win_length=win_size,
        window=hann_window[str(device)],
        center=center, 
        pad_mode='reflect',
        normalized=False, 
        onesided=True,
        return_complex=True
    )
    
    # Magnitude
    spec = torch.abs(spec)
    
    # Energy
    energy = torch.norm(spec, dim=1)
    
    # Mel spectrogram
    mel_output = torch.matmul(mel_basis[str(fmax) + '_' + str(device)], spec)
    
    # Apply log transform
    mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
    
    return mel_output, energy

@processor_registry.register("FeatureProcessor")
class FeatureProcessor(Processor):
    """
    오디오 파일에서 특성(멜 스펙트로그램, 피치, 에너지) 추출 프로세서
    """
    
    def __init__(self, **kwargs):
        """
        초기화
        """
        super().__init__(**kwargs)
        
        # 기본 오디오 매개변수
        self.sampling_rate = 16000
        self.hop_length = 160  # 10ms at 16kHz
        self.win_length = 640  # 40ms at 16kHz
        self.n_fft = 1024
        self.n_mels = 80
        self.fmin = 0
        self.fmax = 8000
        
        # 장치 설정
        self.device = None
        
        self.logger.info("Initialized FeatureProcessor")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        오디오 파일에서 특성 추출
        
        Args:
            context: 처리 컨텍스트
                - 'audio_files': 오디오 파일 정보 리스트
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                
        Returns:
            업데이트된 컨텍스트
                - 'features': 추출된 특성 정보 리스트
        """
        start_time = self._log_start("Starting feature extraction")
        
        # 컨텍스트에서 필요한 정보 가져오기
        audio_files = context.get('audio_files', [])
        output_path = context.get('output_path')
        config = context.get('config')
        
        if not audio_files:
            self.logger.warning("No audio files found in context")
            return context
        
        # 설정 값 가져오기
        self.sampling_rate = config.get('audio.sampling_rate', self.sampling_rate)
        self.hop_length = config.get('audio.hop_length', self.hop_length)
        self.win_length = config.get('audio.win_length', self.win_length)
        self.n_fft = config.get('audio.filter_length', self.n_fft)
        self.n_mels = config.get('audio.n_mel_channels', self.n_mels)
        self.fmin = config.get('audio.mel_fmin', self.fmin)
        self.fmax = config.get('audio.mel_fmax', self.fmax)
        
        # 출력 디렉토리 준비
        mel_dir = os.path.join(output_path, 'preprocessed', 'mel')
        pitch_dir = os.path.join(output_path, 'preprocessed', 'pitch')
        energy_dir = os.path.join(output_path, 'preprocessed', 'energy')
        duration_dir = os.path.join(output_path, 'preprocessed', 'duration')
        
        for directory in [mel_dir, pitch_dir, energy_dir, duration_dir]:
            self._ensure_directory(directory)
        
        # 장치 설정
        cuda_device = config.get('cuda_device', 0) if config.get('device') == 'cuda' else None
        self.device = get_device(cuda_device)
        self.logger.info(f"Using device: {self.device}")
        
        # 특성 정보 리스트
        features = []
        
        # 각 오디오 파일에 대해 특성 추출
        for audio_info in tqdm(audio_files, desc="Extracting features"):
            audio_path = audio_info['path']
            audio_id = audio_info['id']
            
            try:
                # 특성 파일 경로
                mel_path = os.path.join(mel_dir, f"{audio_id}.npy")
                pitch_path = os.path.join(pitch_dir, f"{audio_id}.npy")
                energy_path = os.path.join(energy_dir, f"{audio_id}.npy")
                duration_path = os.path.join(duration_dir, f"{audio_id}.npy")
                
                # 이미 처리된 파일이면 건너뛰기
                if (os.path.exists(mel_path) and os.path.exists(pitch_path) and 
                    os.path.exists(energy_path) and os.path.exists(duration_path)):
                    self.logger.debug(f"Features already extracted for {audio_id}")
                    
                    # 이미 추출된 특성 정보 추가
                    features.append({
                        'id': audio_id,
                        'mel_path': mel_path,
                        'pitch_path': pitch_path,
                        'energy_path': energy_path,
                        'duration_path': duration_path
                    })
                    continue
                
                # 오디오 파일 로드
                wav, _ = librosa.load(audio_path, sr=self.sampling_rate)
                wav = wav.astype(np.float32)
                
                # 피치 추출
                pitch, t = pw.dio(
                    wav.astype(np.float64),
                    self.sampling_rate,
                    frame_period=self.hop_length / self.sampling_rate * 1000
                )
                pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
                pitch = pitch[:len(wav) // self.hop_length]
                
                # 피치가 대부분 0인 경우 (무음 오디오) 건너뛰기
                if np.sum(pitch != 0) <= 1:
                    self.logger.warning(f"Unvoiced audio: {audio_path}")
                    continue
                
                # 오디오 텐서로 변환
                wav_tensor = torch.FloatTensor(wav).to(self.device).unsqueeze(0)
                
                # 멜 스펙트로그램과 에너지 추출
                mel_spectrogram, energy = mel_spectrogram_torch(
                    y=wav_tensor,
                    n_fft=self.n_fft,
                    num_mels=self.n_mels,
                    sampling_rate=self.sampling_rate,
                    hop_size=self.hop_length,
                    win_size=self.win_length,
                    fmin=self.fmin,
                    fmax=self.fmax
                )
                
                # CPU로 이동
                mel_spectrogram = mel_spectrogram.squeeze(0).cpu().numpy()
                energy = energy.squeeze(0).cpu().numpy()
                
                # 길이 조정 (4의 배수로)
                mel_length = mel_spectrogram.shape[1]
                if mel_length % 4 != 0:
                    new_length = mel_length - (mel_length % 4)
                    mel_spectrogram = mel_spectrogram[:, :new_length]
                    
                    if new_length > pitch.shape[0]:
                        pitch = np.pad(pitch, (0, new_length - pitch.shape[0]), 
                                       mode="constant", constant_values=0)
                    else:
                        pitch = pitch[:new_length]
                        
                    if new_length > energy.shape[0]:
                        energy = np.pad(energy, (0, new_length - energy.shape[0]), 
                                       mode="constant", constant_values=0)
                    else:
                        energy = energy[:new_length]
                
                # 최종 특성 저장
                mel_length = mel_spectrogram.shape[1]
                duration = np.array([mel_length], dtype=np.int32)
                
                np.save(duration_path, duration)
                np.save(pitch_path, pitch)
                np.save(energy_path, energy)
                np.save(mel_path, mel_spectrogram.T)  # Transpose for compatibility
                
                # 특성 정보 추가
                features.append({
                    'id': audio_id,
                    'mel_path': mel_path,
                    'pitch_path': pitch_path,
                    'energy_path': energy_path,
                    'duration_path': duration_path,
                    'mel_length': mel_length
                })
                
                self.logger.debug(f"Extracted features for {audio_id}")
                
            except Exception as e:
                self.logger.error(f"Error extracting features for {audio_id}: {e}")
        
        # 컨텍스트에 특성 정보 추가
        context['features'] = features
        
        self._log_end(start_time, f"Extracted features for {len(features)} audio files")
        return context