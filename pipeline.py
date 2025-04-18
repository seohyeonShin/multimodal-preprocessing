#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import time
import importlib
import traceback
from typing import Dict, List, Optional, Any, Type, Set
import torch
from tqdm import tqdm
import concurrent.futures

from core.config import Config
from core.base import Processor, DatabaseAdapter
from core.utils import setup_logging, get_device

# 프로세서 단계 정의
PROCESSOR_STAGES = {
    # 비디오 처리 단계
    "frames": {"number": 1, "type": "video", "dependencies": []},
    "faces": {"number": 2, "type": "video", "dependencies": ["frames"]},
    "landmarks": {"number": 3, "type": "video", "dependencies": ["faces"]},
    "mouth": {"number": 4, "type": "video", "dependencies": ["landmarks"]},
    "lip_embedding": {"number": 5, "type": "video", "dependencies": ["mouth"]},
    "va_feature": {"number": 6, "type": "video", "dependencies": ["frames", "landmarks"]},
    
    # 오디오 처리 단계
    "audio": {"number": 7, "type": "audio", "dependencies": []},
    "textgrid": {"number": 8, "type": "audio", "dependencies": ["audio"]},
    "features": {"number": 9, "type": "audio", "dependencies": ["audio"]},
    "normalize": {"number": 10, "type": "audio", "dependencies": ["features"]}
}

# 프로세서 클래스 매핑
PROCESSOR_CLASSES = {
    "frames": "video.frame_extractor.FrameExtractor",
    "faces": "video.face_detector.FaceDetector",
    "landmarks": "video.landmark_processor.LandmarkProcessor",
    "mouth": "video.mouth_extractor.MouthExtractor",
    "lip_embedding": "video.lip_embedding_extractor.LipEmbeddingExtractor",
    "va_feature": "video.va_feature_extractor.VAFeatureExtractor",
    
    "audio": "audio.audio_extractor.AudioExtractor",
    "textgrid": "audio.textgrid_processor.TextGridProcessor",
    "features": "audio.feature_processor.FeatureProcessor",
    "normalize": "audio.audio_normalizer.AudioNormalizer"
}

class PreprocessingPipeline:
    """데이터 전처리 파이프라인"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config or Config()
        self.processors = []
        self.processed_stages = set()  # 처리된 단계 추적
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_processor(self, processor: Processor, stage_name: str) -> 'PreprocessingPipeline':
        """
        파이프라인에 프로세서 추가
        
        Args:
            processor: 추가할 프로세서
            stage_name: 단계 이름
            
        Returns:
            PreprocessingPipeline: 메서드 체이닝을 위한 self
        """
        if not isinstance(processor, Processor):
            raise TypeError("processor must be an instance of Processor")
        
        # 단계 정보 저장
        processor._stage_name = stage_name
        processor._stage_num = PROCESSOR_STAGES.get(stage_name, {}).get("number", 99)
        
        self.processors.append(processor)
        return self
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        전체 파이프라인 실행 - 배치 처리 방식으로 개선
        
        Args:
            input_path: 입력 디렉토리 경로
            output_path: 출력 디렉토리 경로
            **kwargs: 추가 인자
            
        Returns:
            Dict: 최종 컨텍스트
        """
        start_time = time.time()
        
        # 출력 디렉토리 생성
        os.makedirs(output_path, exist_ok=True)
        
        # 데이터베이스 어댑터 생성
        database_type = kwargs.pop('database_type', 'auto')
        database_adapter,detected_type = DatabaseAdapter.create(database_type, input_path)
        
        # 초기 컨텍스트 설정
        context = {
            'input_path': input_path,
            'output_path': output_path,
            'corpus_path': self.config.get('path.corpus'),
            'config': self.config,
            'database_type':detected_type,
            'database_adapter': database_adapter,
            'start_time': start_time,
            **kwargs
        }
        
        # 프로세서를 단계 번호로 정렬
        self.processors.sort(key=lambda p: getattr(p, '_stage_num', 99))
        
        self.logger.info(f"Starting pipeline with {len(self.processors)} processors")
        self.logger.info(f"Input: {input_path}")
        self.logger.info(f"Output: {output_path}")
        self.logger.info(f"Database type: {database_type}")
        
        # 오디오 처리 단계와 비디오 처리 단계 분리
        audio_processors = [p for p in self.processors if getattr(p, '_stage_name', '') in 
                          ['audio', 'textgrid', 'features', 'normalize']]
        video_processors = [p for p in self.processors if getattr(p, '_stage_name', '') not in 
                          ['audio', 'textgrid', 'features', 'normalize']]
        
        if audio_processors:
            self.logger.info("Starting audio processing pipeline")
            context = self._process_audio_pipeline(audio_processors, context)
        
        if video_processors:
            self.logger.info("Starting video processing pipeline")
            context = self._process_video_pipeline(video_processors, context)
            
        total_duration = time.time() - start_time
        self.logger.info(f"Pipeline completed in {total_duration:.2f} seconds. Processed stages: {', '.join(sorted(self.processed_stages))}")
        
        return context
    
    def _process_audio_pipeline(self, processors: List[Processor], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        오디오 처리 파이프라인 실행
        
        Args:
            processors: 오디오 처리 프로세서 목록
            context: 컨텍스트
            
        Returns:
            Dict[str, Any]: 업데이트된 컨텍스트
        """
        # 1. 오디오 추출 단계 (모든 비디오에서 오디오 추출)
        audio_extractor = next((p for p in processors if getattr(p, '_stage_name', '') == 'audio'), None)
        if audio_extractor:
            try:
                self.logger.info("Step 1: Extracting audio from all videos")
                processor_start_time = time.time()
                context = audio_extractor.process(context)
                self.processed_stages.add('audio')
                processor_duration = time.time() - processor_start_time
                self.logger.info(f"Completed audio extraction in {processor_duration:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error in audio extraction stage:")
                self.logger.error(traceback.format_exc())
                if self.config.get('stop_on_error', False):
                    raise
        
        # 2. TextGrid 처리 단계 (모든 오디오 파일에 대한 MFA 일괄 처리)
        textgrid_processor = next((p for p in processors if getattr(p, '_stage_name', '') == 'textgrid'), None)
        if textgrid_processor and 'audio' in self.processed_stages:
            try:
                self.logger.info("Step 2: Running MFA on all audio files")
                processor_start_time = time.time()
                context = textgrid_processor.process(context)
                self.processed_stages.add('textgrid')
                processor_duration = time.time() - processor_start_time
                self.logger.info(f"Completed TextGrid processing in {processor_duration:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error in TextGrid processing stage:")
                self.logger.error(traceback.format_exc())
                if self.config.get('stop_on_error', False):
                    raise
        
        # 3. 특성 추출 단계 (모든 오디오 파일의 특성 추출)
        feature_processor = next((p for p in processors if getattr(p, '_stage_name', '') == 'features'), None)
        if feature_processor and 'audio' in self.processed_stages:
            try:
                self.logger.info("Step 3: Extracting features from all audio files")
                processor_start_time = time.time()
                context = feature_processor.process(context)
                self.processed_stages.add('features')
                processor_duration = time.time() - processor_start_time
                self.logger.info(f"Completed feature extraction in {processor_duration:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error in feature extraction stage:")
                self.logger.error(traceback.format_exc())
                if self.config.get('stop_on_error', False):
                    raise
        
        # 4. 정규화 단계 (모든 특성 정규화)
        normalizer = next((p for p in processors if getattr(p, '_stage_name', '') == 'normalize'), None)
        if normalizer and 'features' in self.processed_stages:
            try:
                self.logger.info("Step 4: Normalizing all features")
                processor_start_time = time.time()
                context = normalizer.process(context)
                self.processed_stages.add('normalize')
                processor_duration = time.time() - processor_start_time
                self.logger.info(f"Completed normalization in {processor_duration:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error in normalization stage:")
                self.logger.error(traceback.format_exc())
                if self.config.get('stop_on_error', False):
                    raise
        
        return context
    
    def _process_video_pipeline(self, processors: List[Processor], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        비디오 처리 파이프라인 실행
        
        Args:
            processors: 비디오 처리 프로세서 목록
            context: 컨텍스트
            
        Returns:
            Dict[str, Any]: 업데이트된 컨텍스트
        """
        # 각 비디오 프로세서 실행 (의존성 순서대로)
        for idx, processor in enumerate(processors):
            processor_start_time = time.time()
            processor_name = processor.name
            stage_name = getattr(processor, '_stage_name', 'unknown')
            stage_num = getattr(processor, '_stage_num', 99)
            
            # 의존성 검사
            dependencies = PROCESSOR_STAGES.get(stage_name, {}).get("dependencies", [])
            missing_deps = [dep for dep in dependencies if dep not in self.processed_stages]
            
            if missing_deps:
                self.logger.warning(f"Stage {stage_num:02d}-{stage_name} has unmet dependencies: {', '.join(missing_deps)}")
                if not context.get('force', False):
                    self.logger.error(f"Skipping stage {stage_num:02d}-{stage_name} due to unmet dependencies. Use --force to override.")
                    continue
                else:
                    self.logger.warning(f"Forcing execution despite unmet dependencies.")
            
            try:
                self.logger.info(f"[{idx+1}/{len(processors)}] Running stage {stage_num:02d}-{stage_name} ({processor_name})")
                
                # 프로세서 실행
                context = processor.process(context)
                
                # 처리된 단계 추가
                self.processed_stages.add(stage_name)
                
                processor_duration = time.time() - processor_start_time
                self.logger.info(f"Completed stage {stage_num:02d}-{stage_name} in {processor_duration:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error in stage {stage_num:02d}-{stage_name}:")
                self.logger.error(traceback.format_exc())
                
                if self.config.get('stop_on_error', False):
                    raise
                elif not context.get('continue_on_error', False):
                    self.logger.error(f"Stopping pipeline due to error. Use --continue-on-error to continue despite errors.")
                    break
        
        return context

def load_processor_class(class_path: str) -> Type[Processor]:
    """
    클래스 경로에서 프로세서 클래스 로드
    
    Args:
        class_path: 클래스 경로 (예: 'video.frame_extractor.FrameExtractor')
        
    Returns:
        Type[Processor]: 프로세서 클래스
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        processor_class = getattr(module, class_name)
        return processor_class
    except (ImportError, AttributeError) as e:
        logging.error(f"Failed to load processor class {class_path}: {e}")
        raise

def create_selected_pipeline(config: Config, selected_stages: List[str], 
                           force: bool = False) -> PreprocessingPipeline:
    """
    선택된 단계로 파이프라인 생성
    
    Args:
        config: 설정 객체
        selected_stages: 선택된 단계 목록
        force: 의존성 검사 무시 여부
        
    Returns:
        PreprocessingPipeline: 생성된 파이프라인
    """
    pipeline = PreprocessingPipeline(config)
    
    # 선택된 모든 단계와 그 의존성 수집
    required_stages = set()
    
    if not force:
        # 의존성 추가
        for stage in selected_stages:
            required_stages.add(stage)
            deps = PROCESSOR_STAGES.get(stage, {}).get("dependencies", [])
            required_stages.update(deps)
        
        # 단계 번호 순으로 정렬
        sorted_stages = sorted(required_stages, 
                              key=lambda s: PROCESSOR_STAGES.get(s, {}).get("number", 99))
    else:
        # 의존성 무시하고 선택된 단계만 사용
        sorted_stages = sorted(selected_stages, 
                              key=lambda s: PROCESSOR_STAGES.get(s, {}).get("number", 99))
    
    # 프로세서 추가
    for stage_name in sorted_stages:
        class_path = PROCESSOR_CLASSES.get(stage_name)
        if not class_path:
            logging.warning(f"No processor class found for stage '{stage_name}'. Skipping.")
            continue
        
        try:
            processor_class = load_processor_class(class_path)
            processor = processor_class()
            pipeline.add_processor(processor, stage_name)
            logging.debug(f"Added processor for stage {stage_name}: {class_path}")
        except Exception as e:
            logging.error(f"Failed to create processor for stage '{stage_name}': {e}")
            logging.error(traceback.format_exc())
            if config.get('stop_on_error', False):
                raise
    
    return pipeline

def create_pipeline_from_config(config_path: str) -> PreprocessingPipeline:
    """
    설정 파일에서 파이프라인 생성
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        PreprocessingPipeline: 설정에 따른 파이프라인
    """
    config = Config(config_path)
    pipeline = PreprocessingPipeline(config)
    
    # 설정에서 프로세서 목록 가져오기
    processors_config = config.get('processors', [])
    
    if not processors_config:
        # 모든 단계 사용
        all_stages = sorted(PROCESSOR_STAGES.keys(), 
                           key=lambda s: PROCESSOR_STAGES.get(s, {}).get("number", 99))
        return create_selected_pipeline(config, all_stages)
    
    for processor_config in processors_config:
        stage_name = processor_config.get('stage')
        class_path = processor_config.get('class') or PROCESSOR_CLASSES.get(stage_name)
        
        if not class_path:
            logging.warning(f"No processor class found for stage '{stage_name}'. Skipping.")
            continue
        
        try:
            # 프로세서 모듈과 클래스 이름 분리
            module_path, class_name = class_path.rsplit('.', 1)
            
            # 모듈 동적 임포트
            module = importlib.import_module(module_path)
            # 클래스 가져오기
            processor_class = getattr(module, class_name)
            # 인스턴스 생성
            args = processor_config.get('args', {})
            processor = processor_class(**args)
            
            # 파이프라인에 추가
            pipeline.add_processor(processor, stage_name)
        except Exception as e:
            logging.error(f"Failed to create processor for stage '{stage_name}': {e}")
            logging.error(traceback.format_exc())
            if config.get('stop_on_error', False):
                raise
    
    return pipeline

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='모듈식 TTS 전처리 파이프라인')
    
    # 기본 인자 설정
    parser.add_argument('--input', '-i', type=str, default='/home/jovyan/store1/jenny_DATA/LRS3/pretrain',
                        #'/home/jovyan/store1/jenny_DATA/LRS3/pretrain',
                        help='비디오가 있는 입력 디렉토리')
    parser.add_argument('--output', '-o', type=str, default='/home/jovyan/store1/jenny_DATA/Preprocess/LRS3',
                        help='처리된 데이터의 출력 디렉토리')
    parser.add_argument('--corpus-dir', type=str, default=None,
                    help='텍스트 파일과 오디오 파일이 있는 corpus 디렉토리 (지정하지 않으면 출력 디렉토리 내에 자동 생성)')

    parser.add_argument('--config', '-c', type=str, default=None,
                        help='설정 파일 경로')
    parser.add_argument('--log-level', '-l', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='로깅 레벨')
    parser.add_argument('--log-file', type=str, default=None,
                        help='로그 파일 경로')
    
    # 처리 옵션
    parser.add_argument('--stages', type=str, nargs='+',
                        choices=list(PROCESSOR_STAGES.keys()) + ['all', 'video', 'audio'],
                        default=['audio'],
                        help='처리할 단계 선택 (all, video, audio, 또는 개별 단계)')
    parser.add_argument('--force', action='store_true',
                        help='의존성 검사 무시하고 선택한 단계만 실행')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='오류 발생해도 다음 단계 계속 실행')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='오류 발생 시 즉시 종료')
    
    # 데이터베이스 구조 관련 인자
    parser.add_argument('--database-type', type=str, default='auto',
                        choices=['auto', 'lrs3', 'vox', 'standard'],
                        help='데이터베이스 구조 유형')
    
    # 데이터셋 이름 인자
    parser.add_argument('--dataset-name', type=str, default='AnyOne',
                       help='데이터셋/화자 이름')
    
    # 단일 비디오 처리 인자
    parser.add_argument('--input-video', type=str, default=None,
                       help='직접 처리할 단일 비디오 파일 경로')
    
    # 병렬 처리 관련 인자
    parser.add_argument('--max-workers', type=int, default=4,
                       help='병렬 처리에 사용할 최대 워커 수')

    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level, args.log_file)
    
    # 설정 객체 생성
    config = Config(args.config) if args.config else Config()
    
    # 명령행 인자로 설정 업데이트
    config.set('dataset_name', args.dataset_name)
    config.set('stop_on_error', args.stop_on_error)
    config.set('max_workers', args.max_workers)
    
    # 처리 단계 결정
    selected_stages = []
    
    if 'all' in args.stages:
        selected_stages = list(PROCESSOR_STAGES.keys())
    else:
        for stage in args.stages:
            if stage == 'video':
                # 비디오 처리 단계만 추가
                selected_stages.extend([s for s, info in PROCESSOR_STAGES.items() 
                                      if info.get('type') == 'video'])
            elif stage == 'audio':
                # 오디오 처리 단계만 추가
                selected_stages.extend([s for s, info in PROCESSOR_STAGES.items() 
                                      if info.get('type') == 'audio'])
            else:
                # 개별 단계 추가
                selected_stages.append(stage)

    if args.corpus_dir:
        config.set('path.corpus', args.corpus_dir)
        logging.info(f"Using existing corpus directory: {args.corpus_dir}")
    else:
        config.set('path.corpus', os.path.join(args.output, 'corpus'))
        logging.info(f"Using default corpus directory in output: {config.get('path.corpus')}")

    # 중복 제거
    selected_stages = list(dict.fromkeys(selected_stages))
    
    logging.info(f"Selected stages: {', '.join(selected_stages)}")
    
    # 단일 비디오 처리
    if args.input_video:
        import tempfile
        import shutil
        
        # 임시 디렉토리에 비디오 복사
        with tempfile.TemporaryDirectory() as temp_dir:
            video_name = os.path.basename(args.input_video)
            video_path = os.path.join(temp_dir, video_name)
            shutil.copy2(args.input_video, video_path)
            
            # 파이프라인 생성 및 실행
            pipeline = create_selected_pipeline(config, selected_stages, args.force)
            pipeline.process(temp_dir, args.output, database_type='standard',
                           force=args.force, continue_on_error=args.continue_on_error)
            
            logging.info(f"Single video processing completed: {args.input_video}")
    else:
        # 설정 파일에서 파이프라인 생성
        if args.config:
            pipeline = create_pipeline_from_config(args.config)
        else:
            # 선택된 단계로 파이프라인 생성
            pipeline = create_selected_pipeline(config, selected_stages, args.force)
        
        # 파이프라인 실행
        pipeline.process(args.input, args.output, database_type=args.database_type,
                       force=args.force, continue_on_error=args.continue_on_error)
        
        logging.info(f"Processing completed: {args.input}")

if __name__ == "__main__":
    main()