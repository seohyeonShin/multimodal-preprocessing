'''
설정 로딩 및 관리기능 구현
기본 설정값 정의
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import yaml
import logging
from typing import Any, Dict, Optional, Union

class Config:
    """
    설정 관리 클래스
    다양한 형식(json, yaml)의 설정 파일을 로드하고 관리합니다.
    """
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # 기본 설정값
        self.config = {
            'device': 'cuda',
            'cuda_device': 0,
            'output_root': './output',
            'stop_on_error': False,
            'cache_enabled': True,
            'cache_dir': './.cache',
            'video': {
                'extract_frames': True,
                'frame_rate': 25,
                'use_ffmpeg': True,
                'face_detection': {
                    'confidence_threshold': 0.9,
                    'scales': [0.25],
                },
                'landmark_detection': {
                    'use_face_alignment': True,
                    'path_to_detector':'/home/jovyan/store1/TTS_workspace/CODE/dataProcessing/s3fd-619a316812.pth'
                },
                'mouth': {
                    'width': 96,
                    'height': 96,
                    'start_idx': 48,
                    'stop_idx': 68,
                    'window_margin': 12
                }
            },
            'audio': {
                'extract_audio': True,
                'sampling_rate': 16000,
                'hop_length': 160,
                'win_length': 640,
                'filter_length': 1024,
                'n_mel_channels': 80,
                'mel_fmin': 0,
                'mel_fmax': 8000,
                'normalize': {
                    'pitch': True,
                    'energy': True
                },
                'mfa': {
                    'dictionary_path': '/home/jovyan/store1/TTS_workspace/CODE/dataProcessing/lexicon/librispeech-lexicon.txt',
                    'acoustic_model_path': '/home/jovyan/store1/TTS_workspace/CODE/dataProcessing/MFA/acoustic/english_us_arpa.zip',
                    'g2p_model_path': '/home/jovyan/store1/TTS_workspace/CODE/dataProcessing/MFA/g2p/english_us_arpa.zip',
                }
            },
            'val_size': 100
        }
        
        # 환경변수 설정
        self._apply_environment_variables()
        
        # 설정 파일에서 로드
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
            
        # 인자로 전달된 설정으로 덮어쓰기
        self._update_config(kwargs)
        
        # 설정 검증
        self._validate_config()
        
        # 설정 로그 출력
        self._log_config_summary()
    
    def _apply_environment_variables(self):
        """환경변수에서 설정 값 가져오기"""
        # CUDA 장치 설정
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            try:
                self.config['cuda_device'] = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
            except (ValueError, IndexError):
                pass
        
        # 출력 디렉토리 설정
        if 'OUTPUT_ROOT' in os.environ:
            self.config['output_root'] = os.environ['OUTPUT_ROOT']
    
    def _load_config(self, config_path: str):
        """설정 파일에서 설정 로드"""
        ext = os.path.splitext(config_path)[1].lower()
        try:
            if ext == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            elif ext in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
                
            self._update_config(config_data)
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def _update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        def _recursive_update(d: Dict[str, Any], u: Dict[str, Any]):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _recursive_update(d[k], v)
                else:
                    d[k] = v
        
        _recursive_update(self.config, new_config)
    
    def _validate_config(self):
        """설정 값 검증"""
        # 필수 설정 확인
        required_keys = []
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Required configuration key '{key}' is missing")
        
        # 값 범위 확인
        if self.get('audio.sampling_rate') <= 0:
            raise ValueError("Sampling rate must be positive")
        
        if self.get('video.frame_rate') <= 0:
            raise ValueError("Frame rate must be positive")
    
    def _log_config_summary(self):
        """중요 설정값 로그로 요약"""
        logging.info("Configuration summary:")
        logging.info(f"  Device: {self.get('device')}")
        logging.info(f"  Output root: {self.get('output_root')}")
        logging.info(f"  Video frame rate: {self.get('video.frame_rate')}")
        logging.info(f"  Audio sampling rate: {self.get('audio.sampling_rate')}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값 가져오기
        점 표기법으로 중첩된 설정 값에 접근할 수 있습니다 (예: 'audio.sampling_rate')
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        설정값 설정하기
        점 표기법으로 중첩된 설정 값에 접근할 수 있습니다 (예: 'audio.sampling_rate')
        """
        keys = key.split('.')
        target = self.config
        
        # 마지막 키를 제외한 모든 키에 대해 dict 구조 확인
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            elif not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        
        # 마지막 키에 값 설정
        target[keys[-1]] = value
    
    def save(self, config_path: str) -> None:
        """현재 설정을 파일로 저장"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if ext == '.json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            elif ext in ['.yaml', '.yml']:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
            
            logging.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logging.error(f"Error saving config to {config_path}: {e}")
            raise