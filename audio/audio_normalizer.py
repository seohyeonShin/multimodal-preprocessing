#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import json
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from core.base import Processor, processor_registry

@processor_registry.register("AudioNormalizer")
class AudioNormalizer(Processor):
    """
    피치와 에너지 특성을 정규화하는 프로세서
    """
    
    def __init__(self, **kwargs):
        """
        초기화
        """
        super().__init__(**kwargs)
        
        # 정규화 설정
        self.pitch_normalization = True
        self.energy_normalization = True
        
        self.logger.info("Initialized AudioNormalizer")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        피치와 에너지 특성 정규화
        
        Args:
            context: 처리 컨텍스트
                - 'features': 특성 정보 리스트
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                
        Returns:
            업데이트된 컨텍스트
                - 'stats': 정규화 통계 정보
        """
        start_time = self._log_start("Starting audio feature normalization")
        
        # 컨텍스트에서 필요한 정보 가져오기
        features = context.get('features', [])
        output_path = context.get('output_path')
        config = context.get('config')
        
        if not features:
            self.logger.warning("No features found in context")
            return context
        
        # 설정 값 가져오기
        self.pitch_normalization = config.get('audio.normalize.pitch', self.pitch_normalization)
        self.energy_normalization = config.get('audio.normalize.energy', self.energy_normalization)
        
        # 출력 디렉토리 설정
        preprocessed_dir = os.path.join(output_path, 'preprocessed')
        pitch_dir = os.path.join(preprocessed_dir, 'pitch')
        energy_dir = os.path.join(preprocessed_dir, 'energy')
        
        # 통계 계산을 위한 스케일러 초기화
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        
        # 모든 특성 파일에서 통계 계산
        self.logger.info("Computing statistics from features")
        for feature_info in tqdm(features, desc="Computing statistics"):
            pitch_path = feature_info.get('pitch_path')
            energy_path = feature_info.get('energy_path')
            
            if pitch_path and os.path.exists(pitch_path):
                pitch_values = np.load(pitch_path)
                pitch_values = self._remove_outliers(pitch_values)
                
                if len(pitch_values) > 0:
                    pitch_scaler.partial_fit(pitch_values.reshape((-1, 1)))
            
            if energy_path and os.path.exists(energy_path):
                energy_values = np.load(energy_path)
                energy_values = self._remove_outliers(energy_values)
                
                if len(energy_values) > 0:
                    energy_scaler.partial_fit(energy_values.reshape((-1, 1)))
        
        # 통계값 계산
        pitch_mean = pitch_scaler.mean_[0] if self.pitch_normalization else 0
        pitch_std = pitch_scaler.scale_[0] if self.pitch_normalization else 1
        energy_mean = energy_scaler.mean_[0] if self.energy_normalization else 0
        energy_std = energy_scaler.scale_[0] if self.energy_normalization else 1
        
        # 정규화 수행
        self.logger.info("Normalizing features")
        pitch_min, pitch_max = self._normalize_dir(pitch_dir, pitch_mean, pitch_std)
        energy_min, energy_max = self._normalize_dir(energy_dir, energy_mean, energy_std)
        
        # 통계 정보 저장
        stats = {
            "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
            "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
        }
        
        stats_path = os.path.join(preprocessed_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        # 스피커 정보 저장 (단일 스피커 가정)
        speakers = {config.get('dataset_name', 'speaker'): 0}
        speakers_path = os.path.join(preprocessed_dir, "speakers.json")
        with open(speakers_path, "w") as f:
            json.dump(speakers, f, indent=2)
        
        # 컨텍스트에 통계 정보 추가
        context['stats'] = stats
        
        self._log_end(start_time, "Feature normalization completed")
        return context
    
    def _normalize_dir(self, directory: str, mean: float, std: float) -> Tuple[float, float]:
        """
        디렉토리 내 모든 특성 파일 정규화
        
        Args:
            directory: 특성 파일이 있는 디렉토리
            mean: 평균값
            std: 표준편차
            
        Returns:
            Tuple[float, float]: (최소값, 최대값)
        """
        if not os.path.exists(directory):
            self.logger.warning(f"Directory does not exist: {directory}")
            return 0.0, 0.0
        
        max_value = float('-inf')
        min_value = float('inf')
        
        for filename in os.listdir(directory):
            if not filename.endswith('.npy'):
                continue
                
            filepath = os.path.join(directory, filename)
            values = np.load(filepath)
            
            # 정규화
            normalized_values = (values - mean) / std
            np.save(filepath, normalized_values)
            
            # 최소값/최대값 업데이트
            if len(normalized_values) > 0:
                file_max = np.max(normalized_values)
                file_min = np.min(normalized_values)
                
                max_value = max(max_value, file_max)
                min_value = min(min_value, file_min)
        
        return min_value, max_value
    
    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """
        특성 값에서 이상치 제거
        
        Args:
            values: 특성 값 배열
            
        Returns:
            np.ndarray: 이상치가 제거된 배열
        """
        values = values[values != 0]  # 0 값 제거
        
        if len(values) == 0:
            return np.array([])
        
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        
        return values[(values > lower) & (values < upper)]