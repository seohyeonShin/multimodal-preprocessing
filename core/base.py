'''
기본 프로세스 인터페이스 정의 
파이프라인 추상 클래스 정의
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import os
import logging
import time
from typing import Dict, Any, List, Optional, TypeVar, Generic, Type

# 제네릭 타입 정의
T = TypeVar('T')

class Processor(ABC):
    """
    모든 프로세서의 기본 인터페이스
    각 전처리 단계를 담당하는 클래스는 이 클래스를 상속해야 함
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)
    
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        컨텍스트를 처리하고 업데이트된 컨텍스트 반환
        
        Args:
            context (dict): 처리 컨텍스트 (입력/출력 경로, 설정 등)
            
        Returns:
            dict: 업데이트된 컨텍스트
        """
        pass
    
    def _ensure_directory(self, path: str) -> None:
        """디렉토리가 존재하는지 확인하고 없으면 생성"""
        os.makedirs(path, exist_ok=True)
        
    def _log_start(self, message: str = "Starting processing") -> float:
        """프로세싱 시작 로깅 및 시작 시간 반환"""
        start_time = time.time()
        self.logger.info(f"{message}")
        return start_time
    
    def _log_end(self, start_time: float, message: str = "Processing completed") -> None:
        """프로세싱 종료 로깅"""
        duration = time.time() - start_time
        self.logger.info(f"{message} in {duration:.2f} seconds")

class DatabaseAdapter(ABC):
    """
    데이터베이스 구조 어댑터 인터페이스
    데이터베이스 구조에 따라 파일 접근 방식을 추상화
    """
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def get_video_files(self) -> List[Dict[str, str]]:
        """
        처리할 비디오 파일 목록 반환
        
        Returns:
            List[Dict[str, str]]: 비디오 파일 정보 목록
                각 항목은 다음 키를 포함:
                - path: 비디오 파일 경로
                - id: 비디오 식별자
                - folder: 비디오가 위치한 폴더 이름
                - index: 폴더 내 비디오 인덱스
        """
        pass
    
    @abstractmethod
    def get_audio_files(self) -> List[Dict[str, str]]:
        """
        처리할 오디오 파일 목록 반환
        
        Returns:
            List[Dict[str, str]]: 오디오 파일 정보 목록
                각 항목은 다음 키를 포함:
                - path: 오디오 파일 경로
                - id: 오디오 식별자
                - transcript_path: 대본 파일 경로 (있는 경우)
        """
        pass
    
    @abstractmethod
    def get_output_path(self, file_info: Dict[str, str], category: str) -> str:
        """
        주어진 파일에 대한 출력 경로 반환
        
        Args:
            file_info: 파일 정보 딕셔너리
            category: 출력 카테고리 (예: 'frames', 'mel', 등)
            
        Returns:
            str: 출력 경로
        """
        pass
    def find_matching_text(self, video_info: Dict[str, str], corpus_dir: str) -> Optional[str]:
        """
        비디오에 대응하는 텍스트 파일 찾기
        
        Args:
            video_info: 비디오 정보
            corpus_dir: corpus 디렉토리 경로
            
        Returns:
            Optional[str]: 텍스트 파일 경로 또는 None
        """
        video_id = video_info['id']
        folder = video_info.get('folder', '')
        index = video_info.get('index', '')
        
        # 가능한 텍스트 파일 패턴들
        patterns = [
            # 1. video_id.txt - 직접 매칭
            os.path.join(corpus_dir, f"{video_id}.txt"),
            # 2. index.txt - LRS3 스타일
            os.path.join(corpus_dir, f"{index}.txt"),
            # 3. video_id_{index}.txt - Standard 스타일
            os.path.join(corpus_dir, f"{folder}_{index}.txt"),
            # 4. folder/index.txt - 원본 위치
            os.path.join(self.root_path, folder, f"{index}.txt")
        ]
        
        # 패턴 순서대로 검색
        for pattern in patterns:
            if os.path.exists(pattern):
                self.logger.debug(f"Found matching text file: {pattern} for video: {video_id}")
                return pattern
        
        self.logger.warning(f"No matching text file found for video: {video_id}")
        return None
    @staticmethod
    def create(database_type: str, root_path: str) -> 'DatabaseAdapter':
        """
        데이터베이스 타입에 따라 적절한 어댑터 생성
        
        Args:
            database_type: 데이터베이스 타입 ('lrs3', 'vox', 'standard', 'auto')
            root_path: 루트 디렉토리 경로
            
        Returns:
            DatabaseAdapter: 적절한 데이터베이스 어댑터
        """
        from core.database import (
            StandardDatabaseAdapter, LRS3DatabaseAdapter, 
            VoxDatabaseAdapter, detect_database_type
        )
        
        if database_type == 'auto':
            database_type = detect_database_type(root_path)

        if database_type == 'lrs3':
            return LRS3DatabaseAdapter(root_path),database_type
        elif database_type == 'vox':
            return VoxDatabaseAdapter(root_path),database_type
        else:  # 'standard' 또는 기타
            return StandardDatabaseAdapter(root_path),database_type

class Registry(Generic[T]):
    """
    클래스 레지스트리
    프로세서나 어댑터 등의 클래스를 이름으로 등록하고 조회할 수 있게 함
    """
    def __init__(self):
        self._registry = {}
    
    def register(self, name: str = None):
        """
        클래스를 레지스트리에 등록하는 데코레이터
        
        Args:
            name: 등록할 이름 (없으면 클래스 이름 사용)
        """
        def inner_wrapper(wrapped_class: Type[T]) -> Type[T]:
            key = name or wrapped_class.__name__
            if key in self._registry:
                logging.warning(f"Class {key} already exists in registry")
            self._registry[key] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    def get(self, name: str) -> Type[T]:
        """
        이름으로 클래스 조회
        
        Args:
            name: 등록된 클래스 이름
            
        Returns:
            등록된 클래스
        
        Raises:
            KeyError: 등록되지 않은 이름
        """
        if name not in self._registry:
            raise KeyError(f"Class {name} not registered")
        return self._registry[name]
    
    def list(self) -> List[str]:
        """
        등록된 모든 클래스 이름 목록 반환
        
        Returns:
            List[str]: 등록된 클래스 이름 목록
        """
        return list(self._registry.keys())

# 전역 레지스트리 인스턴스 생성
processor_registry = Registry[Processor]()