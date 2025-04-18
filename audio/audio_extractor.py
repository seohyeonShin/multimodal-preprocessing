#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path
import logging
from typing import Dict, Any, List
import shutil
from tqdm import tqdm
import concurrent.futures
import glob
import time
from core.base import Processor, processor_registry
from core.utils import run_command

@processor_registry.register("AudioExtractor")
class AudioExtractor(Processor):
    """
    비디오 파일에서 오디오 추출 프로세서
    모든 비디오를 한 번에 처리하도록 개선 및 디버깅 강화
    """
    
    def __init__(self, sampling_rate: int = 16000, mono: bool = True, max_workers: int = 4, **kwargs):
        """
        초기화
        
        Args:
            sampling_rate: 추출된 오디오의 샘플링 레이트 (Hz)
            mono: 모노 오디오로 추출할지 여부
            max_workers: 병렬 처리 시 사용할 최대 워커 수
        """
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.max_workers = max_workers
        self.logger.info(f"Initialized AudioExtractor (sampling_rate={sampling_rate}, mono={mono}, max_workers={max_workers})")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        비디오 파일에서 오디오 추출 - 디버깅을 강화한 버전
        
        Args:
            context: 처리 컨텍스트
                - 'input_path': 입력 디렉토리 경로
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                - 'database_adapter': 데이터베이스 어댑터
                
        Returns:
            업데이트된 컨텍스트
                - 'audio_files': 추출된 오디오 파일 정보 리스트
        """
        start_time = self._log_start("Starting audio extraction")
        
        # 컨텍스트에서 필요한 정보 가져오기
        input_path = context.get('input_path')
        output_path = context.get('output_path')
        database_adapter = context.get('database_adapter')
        config = context.get('config')
        
        # 디버그 정보 출력
        self.logger.info(f"Input path: {input_path}")
        self.logger.info(f"Output path: {output_path}")
        
        # 설정 값 가져오기
        self.sampling_rate = config.get('audio.sampling_rate', self.sampling_rate)
        self.max_workers = config.get('max_workers', self.max_workers)
        self.logger.info(f"Using sampling rate: {self.sampling_rate}")
        self.logger.info(f"Using max workers: {self.max_workers}")
        
        # 출력 디렉토리 준비
        if context.get('corpus_path'):
            corpus_dir = context.get('corpus_path')
        else:
            corpus_dir = os.path.join(output_path, 'corpus')
        
        output_corpus_dir = os.path.join(output_path, 'corpus')
        
        self.logger.info(f"Corpus directory: {corpus_dir}")
        self.logger.info(f"Output corpus directory: {output_corpus_dir}")

        # 디렉토리 생성 및 확인
        self._ensure_directory(corpus_dir)
        self._ensure_directory(output_corpus_dir)
        
        if not os.path.exists(output_corpus_dir):
            self.logger.error(f"Failed to create directory: {output_corpus_dir}")
            raise RuntimeError(f"Cannot create output directory: {output_corpus_dir}")
        else:
            self.logger.info(f"Output corpus directory exists: {output_corpus_dir}")
        
        # 처리할 파일 목록 가져오기 (database_adapter에서 직접 가져옴)
        video_files = []
        
        # 먼저 컨텍스트에서 확인
        if 'video_files' in context and context['video_files']:
            video_files = context['video_files']
            self.logger.info(f"Using {len(video_files)} video files from context")
        else:
            # 데이터베이스 어댑터에서 가져오기
            try:
                video_files = database_adapter.get_video_files()
                self.logger.info(f"Found {len(video_files)} video files from database adapter")
                
                # 디버그: 처음 몇개 비디오 파일 정보 출력
                for i, video in enumerate(video_files[:3]):
                    self.logger.info(f"Video {i+1}: {video.get('id')} at {video.get('path')}")
                
                # 컨텍스트에 저장
                context['video_files'] = video_files
            except Exception as e:
                self.logger.error(f"Error getting video files from database adapter: {e}")
                # 백업 방법: 직접 디렉토리에서 파일 찾기
                self.logger.info(f"Trying to find video files directly in {input_path}")
                mp4_files = glob.glob(os.path.join(input_path, "**/*.mp4"), recursive=True)
                
                for mp4_file in mp4_files:
                    video_id = os.path.splitext(os.path.basename(mp4_file))[0]
                    video_files.append({
                        'path': mp4_file,
                        'id': video_id,
                        'folder': 'backup',
                        'index': video_id
                    })
                self.logger.info(f"Found {len(video_files)} video files directly")
        
        if not video_files:
            self.logger.warning("No video files found to process")
            return context
        
        # 오디오 파일 목록
        audio_files = []
        
        # 이미 처리된 파일 확인
        existing_wav_files = glob.glob(os.path.join(output_corpus_dir, "*.wav"))
        existing_ids = [os.path.splitext(os.path.basename(wav))[0] for wav in existing_wav_files]
        self.logger.info(f"Found {len(existing_wav_files)} existing WAV files in output directory")
        
        # 처리할 파일 필터링
        extraction_tasks = []
        for video_info in video_files:
            video_path = video_info.get('path')
            video_id = video_info.get('id')
            
            if not video_path or not os.path.exists(video_path):
                self.logger.warning(f"Video file not found: {video_path} for ID {video_id}")
                continue
                
            output_wav = os.path.join(output_corpus_dir, f"{video_id}.wav")
            
            # 이미 추출된 파일은 추가만 하고 건너뛰기
            if video_id in existing_ids:
                self.logger.debug(f"Audio already extracted for {video_id}")
                audio_files.append({
                    'path': output_wav,
                    'id': video_id,
                    'sampling_rate': self.sampling_rate,
                    'source_video': video_path
                })
                continue
            
            # 추출 작업 추가
            extraction_tasks.append((video_info, output_wav))
        
        self.logger.info(f"Need to extract audio from {len(extraction_tasks)} video files")
        
        # 병렬 처리로 오디오 추출
        if extraction_tasks:
            processed_count = 0
            
            # 병렬 처리 사용
            if self.max_workers > 1:
                self.logger.info(f"Using parallel processing with {self.max_workers} workers")
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        future_to_video = {
                            executor.submit(self._extract_audio, task[0], task[1]): task[0].get('id') 
                            for task in extraction_tasks
                        }
                        
                        for future in tqdm(concurrent.futures.as_completed(future_to_video), 
                                        total=len(extraction_tasks),
                                        desc="Extracting audio"):
                            video_id = future_to_video[future]
                            try:
                                result = future.result()
                                if result:
                                    audio_files.append(result)
                                    processed_count += 1
                            except Exception as e:
                                self.logger.error(f"Error extracting audio from {video_id}: {e}")
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {e}")
                    self.logger.info("Falling back to sequential processing")
                    for task in tqdm(extraction_tasks, desc="Extracting audio (sequential)"):
                        try:
                            result = self._extract_audio(task[0], task[1])
                            if result:
                                audio_files.append(result)
                                processed_count += 1
                        except Exception as e:
                            self.logger.error(f"Error extracting audio: {e}")
            else:
                # 순차 처리
                self.logger.info("Using sequential processing")
                for task in tqdm(extraction_tasks, desc="Extracting audio"):
                    try:
                        result = self._extract_audio(task[0], task[1])
                        if result:
                            audio_files.append(result)
                            processed_count += 1
                    except Exception as e:
                        self.logger.error(f"Error extracting audio: {e}")
            
            self.logger.info(f"Successfully extracted audio from {processed_count}/{len(extraction_tasks)} videos")
            
            # 디렉토리 내용 확인
            wav_files = glob.glob(os.path.join(output_corpus_dir, "*.wav"))
            self.logger.info(f"Total WAV files in output directory after extraction: {len(wav_files)}")
        
        # 결과 확인
        if not audio_files:
            self.logger.warning("No audio files were extracted or found")
        else:
            self.logger.info(f"Total audio files in result: {len(audio_files)}")
        
        # 텍스트 파일 준비
        self.logger.info("Preparing transcript files")
        transcript_count = 0
        
        for audio_info in tqdm(audio_files, desc="Preparing transcripts"):
            video_info = next((v for v in video_files if v.get('id') == audio_info.get('id')), None)
            if video_info:
                try:
                    self._prepare_transcript(audio_info, video_info, corpus_dir, output_corpus_dir)
                    transcript_count += 1
                except Exception as e:
                    self.logger.error(f"Error preparing transcript for {audio_info.get('id')}: {e}")
        
        self.logger.info(f"Prepared {transcript_count} transcript files")
        
        # 오디오 파일 정보를 컨텍스트에 저장
        context['audio_files'] = audio_files
        
        self._log_end(start_time, f"Extracted audio from {len(audio_files)} video files")
        return context
    
    def _extract_audio(self, video_info: Dict[str, Any], output_wav: str) -> Dict[str, Any]:
        """
        단일 비디오에서 오디오 추출
        
        Args:
            video_info: 비디오 정보
            output_wav: 출력 WAV 파일 경로
            
        Returns:
            Dict[str, Any]: 오디오 파일 정보 또는 None (실패시)
        """
        video_path = video_info.get('path')
        video_id = video_info.get('id')
        
        if not video_path or not os.path.exists(video_path):
            self.logger.warning(f"Video file not found: {video_path} for ID {video_id}")
            return None
        
        # FFmpeg 명령 구성
        audio_channels = 1 if self.mono else 2
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.sampling_rate), 
            '-ac', str(audio_channels),
            output_wav
        ]
        
        # 명령 실행
        try:
            returncode, stdout, stderr = run_command(command, silent=True)
            
            if returncode != 0:
                self.logger.warning(f"FFmpeg failed for {video_id}: {stderr}")
                return None
            
            # 파일 크기 확인
            if not os.path.exists(output_wav):
                self.logger.warning(f"Output file not created: {output_wav}")
                return None
                
            if os.path.getsize(output_wav) == 0:
                self.logger.warning(f"Extracted audio file is empty: {output_wav}")
                os.remove(output_wav)  # 빈 파일 제거
                return None
            
            # 성공적으로 추출된 경우
            self.logger.debug(f"Successfully extracted audio for {video_id}")
            return {
                'path': output_wav,
                'id': video_id,
                'sampling_rate': self.sampling_rate,
                'source_video': video_path
            }
        except Exception as e:
            self.logger.error(f"Exception during audio extraction for {video_id}: {e}")
            return None
    
    def _prepare_transcript(self, audio_info: Dict[str, Any], video_info: Dict[str, Any], 
                        corpus_dir: str, output_corpus_dir: str) -> None:
        """
        비디오 파일에 대응하는 텍스트 파일 처리
        LRS3 형식 텍스트 파일은 대본을 추출하여 표준 형식으로 저장
        
        Args:
            audio_info: 오디오 파일 정보
            video_info: 비디오 파일 정보
            corpus_dir: 원본 corpus 디렉토리 경로
            output_corpus_dir: 출력 corpus 디렉토리 경로
        """
        video_path = video_info.get('path')
        video_id = video_info.get('id')
        folder = video_info.get('folder', '')
        index = video_info.get('index', '')
        
        # 출력 텍스트 파일 경로
        output_txt = os.path.join(output_corpus_dir, f"{video_id}.txt")
        corpus_txt = os.path.join(corpus_dir, f"{video_id}.txt")

        # 이미 지정한 corpus_dir에 텍스트 파일이 있으면 output path corpus에 옮기기 
        if os.path.exists(corpus_txt):
            shutil.copy2(corpus_txt, output_txt)
            self.logger.debug(f"Copied existing transcript from corpus: {corpus_txt} to {output_txt}")
            return
        
        # 가능한 원본 텍스트 파일 위치들
        potential_txt_paths = [
            # 1. 비디오와 같은 디렉토리의 같은 이름 텍스트 파일
            os.path.join(os.path.dirname(video_path), f"{os.path.splitext(os.path.basename(video_path))[0]}.txt"),
            # 2. 폴더/인덱스.txt (LRS3 스타일)
            os.path.join(os.path.dirname(os.path.dirname(video_path)), folder, f"{index}.txt"),
            # 3. 기타 다양한 패턴
            os.path.join(corpus_dir, f"{folder}_{index}.txt"),
            os.path.join(corpus_dir, f"{index}.txt")
        ]
        
        # 원본 텍스트 파일 찾기
        found_txt = None
        for txt_path in potential_txt_paths:
            if os.path.exists(txt_path):
                found_txt = txt_path
                break
        
        if found_txt:
            try:
                # 텍스트 형식 확인
                is_lrs3_format = self._is_lrs3_format(found_txt)
                
                if is_lrs3_format:
                    # LRS3 형식인 경우 대본 추출 후 표준 형식으로 저장
                    transcript = self._extract_lrs3_transcript(found_txt)
                    with open(output_txt, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                    self.logger.debug(f"Extracted transcript from LRS3 format: {found_txt} to {output_txt}")
                else:
                    # 표준 형식인 경우 그대로 복사
                    shutil.copy2(found_txt, output_txt)
                    self.logger.debug(f"Copied transcript from {found_txt} to {output_txt}")
            except Exception as e:
                self.logger.warning(f"Failed to process transcript for {video_id}: {e}")
        else:
            # 원본 텍스트 파일이 없으면 로그 출력
            self.logger.warning(f"No transcript found for {video_id}")

    def _is_lrs3_format(self, text_path: str) -> bool:
        """
        텍스트 파일이 LRS3 형식인지 확인
        
        Args:
            text_path: 텍스트 파일 경로
            
        Returns:
            bool: LRS3 형식 여부
        """
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read(200)  # 처음 200자만 읽기
            
            # LRS3 형식 확인 (Text: ... Conf: ... 또는 WORD START END 패턴)
            return 'Text:' in content and ('Conf:' in content or 'WORD START END' in content)
        except Exception as e:
            self.logger.error(f"Error checking LRS3 format: {e}")
            return False

    def _extract_lrs3_transcript(self, text_path: str) -> str:
        """
        LRS3 형식 텍스트 파일에서 대본 추출
        
        Args:
            text_path: LRS3 텍스트 파일 경로
            
        Returns:
            str: 추출된 대본
        """
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            import re
            transcript_match = re.search(r'Text:\s+(.*?)(?:\s+Conf:|$)', content, re.DOTALL)
            if transcript_match:
                return transcript_match.group(1).strip()
            else:
                self.logger.warning(f"Could not extract transcript from LRS3 file: {text_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Error extracting LRS3 transcript: {e}")
            return ""