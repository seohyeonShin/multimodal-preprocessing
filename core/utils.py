'''
공통으로 사용되는 유틸리티 함수 구현 
-로깅
-파일처리 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import hashlib
import logging
import time
import shutil
from pathlib import Path
import numpy as np
import torch
import subprocess
from typing import Optional, List, Dict, Any, Union, Tuple

def setup_logging(log_level: str = "info", log_file: Optional[str] = None) -> None:
    """
    로깅 설정
    
    Args:
        log_level: 로그 레벨 ('debug', 'info', 'warning', 'error')
        log_file: 로그 파일 경로 (없으면 콘솔에만 출력)
    """
    level = getattr(logging, log_level.upper())
    
    handlers = []
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # 파일 핸들러 (지정된 경우)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 로거 설정
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )

def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    파일의 해시값 계산
    
    Args:
        file_path: 해시값을 계산할 파일 경로
        algorithm: 해시 알고리즘 ('md5', 'sha1', 'sha256')
        
    Returns:
        str: 파일의 해시값
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def get_device(cuda_device: Optional[int] = None) -> torch.device:
    """
    계산에 사용할 장치 반환
    
    Args:
        cuda_device: 사용할 CUDA 장치 인덱스 (없으면 자동 선택)
        
    Returns:
        torch.device: 계산에 사용할 장치
    """
    if torch.cuda.is_available():
        if cuda_device is not None:
            return torch.device(f"cuda:{cuda_device}")
        return torch.device("cuda")
    return torch.device("cpu")

def run_command(command: List[str], silent: bool = False) -> Tuple[int, str, str]:
    """
    외부 명령 실행
    
    Args:
        command: 실행할 명령과 인자들의 리스트
        silent: 출력을 표시하지 않을지 여부
        
    Returns:
        Tuple[int, str, str]: (반환 코드, 표준 출력, 표준 오류)
    """
    if not silent:
        logging.info(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    
    if process.returncode != 0 and not silent:
        logging.error(f"Command failed with code {process.returncode}")
        logging.error(f"Stderr: {stderr}")
    
    return process.returncode, stdout, stderr

class CacheManager:
    """파일 캐싱 관리자"""
    
    def __init__(self, cache_dir: str = './.cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str, extension: str = '') -> str:
        """
        캐시 키에 대한 파일 경로 생성
        
        Args:
            key: 캐시 키
            extension: 파일 확장자
            
        Returns:
            str: 캐시 파일 경로
        """
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}{extension}")
    
    def has_cache(self, key: str, extension: str = '') -> bool:
        """
        캐시 존재 여부 확인
        
        Args:
            key: 캐시 키
            extension: 파일 확장자
            
        Returns:
            bool: 캐시 존재 여부
        """
        return os.path.exists(self.get_cache_path(key, extension))
    
    def get_cache(self, key: str, extension: str = '') -> Optional[str]:
        """
        캐시 파일 경로 반환 (없으면 None)
        
        Args:
            key: 캐시 키
            extension: 파일 확장자
            
        Returns:
            Optional[str]: 캐시 파일 경로 또는 None
        """
        cache_path = self.get_cache_path(key, extension)
        return cache_path if os.path.exists(cache_path) else None
    
    def save_cache(self, key: str, data_path: str, extension: str = '') -> str:
        """
        파일을 캐시에 저장
        
        Args:
            key: 캐시 키
            data_path: 원본 파일 경로
            extension: 파일 확장자
            
        Returns:
            str: 캐시 파일 경로
        """
        cache_path = self.get_cache_path(key, extension)
        shutil.copy2(data_path, cache_path)
        return cache_path
    
    def save_numpy_cache(self, key: str, data: np.ndarray) -> str:
        """
        NumPy 배열을 캐시에 저장
        
        Args:
            key: 캐시 키
            data: 저장할 NumPy 배열
            
        Returns:
            str: 캐시 파일 경로
        """
        cache_path = self.get_cache_path(key, '.npy')
        np.save(cache_path, data)
        return cache_path
    
    def load_numpy_cache(self, key: str) -> Optional[np.ndarray]:
        """
        NumPy 배열을 캐시에서 로드
        
        Args:
            key: 캐시 키
            
        Returns:
            Optional[np.ndarray]: 로드된 NumPy 배열 또는 None
        """
        cache_path = self.get_cache_path(key, '.npy')
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None
    
    def clear_cache(self) -> None:
        """캐시 디렉토리의 모든 파일 삭제"""
        if os.path.exists(self.cache_dir):
            for file_name in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
    
    def get_cache_size(self) -> int:
        """
        캐시 디렉토리의 총 크기 계산 (바이트)
        
        Returns:
            int: 총 캐시 크기 (바이트)
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

def time_it(func):
    """
    함수의 실행 시간을 측정하는 데코레이터
    
    Args:
        func: 실행 시간을 측정할 함수
        
    Returns:
        함수의 실행 시간을 로깅하는 래퍼 함수
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 함수명과 실행 시간 로깅
        logging.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper