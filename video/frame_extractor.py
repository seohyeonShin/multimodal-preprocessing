#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import subprocess
from tqdm import tqdm
from typing import Dict, Any

from core.base import Processor, processor_registry
from core.utils import run_command
# 1. 
@processor_registry.register("FrameExtractor")
class FrameExtractor(Processor):
    """
    비디오에서 프레임 추출하는 프로세서
    """
    
    def __init__(self, frame_rate: int = 25, use_ffmpeg: bool = True, **kwargs):
        """
        초기화
        
        Args:
            frame_rate: 추출할 프레임 레이트
            use_ffmpeg: FFmpeg 사용 여부 (False면 OpenCV 사용)
        """
        super().__init__(**kwargs)
        self.frame_rate = frame_rate
        self.use_ffmpeg = use_ffmpeg
        self.logger.info(f"Initialized FrameExtractor (frame_rate={frame_rate}, use_ffmpeg={use_ffmpeg})")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        비디오에서 프레임 추출
        
        Args:
            context: 처리 컨텍스트
                - 'input_path': 입력 디렉토리 경로
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                - 'database_adapter': 데이터베이스 어댑터
                
        Returns:
            업데이트된 컨텍스트
                - 'video_files': 처리된 비디오 파일 정보 리스트
        """
        start_time = self._log_start("Starting frame extraction")
        
        # 컨텍스트에서 필요한 정보 가져오기
        input_path = context.get('input_path')
        output_path = context.get('output_path')
        database_adapter = context.get('database_adapter')
        config = context.get('config')
        
        # 설정 값 가져오기
        self.frame_rate = config.get('video.frame_rate', self.frame_rate)
        self.use_ffmpeg = config.get('video.use_ffmpeg', self.use_ffmpeg)
        
        # 출력 디렉토리 (frames) 준비
        frames_root = os.path.join(output_path, 'frames')
        self._ensure_directory(frames_root)
        
        # 처리할 비디오 파일 목록 가져오기
        video_files = database_adapter.get_video_files()
        self.logger.info(f"Found {len(video_files)} video files")
        
        # 각 비디오 파일에 대해 프레임 추출
        processed_videos = []
        
        for video_info in tqdm(video_files, desc="Extracting frames"):
            video_path = video_info['path']
            video_id = video_info['id']
            
            # 출력 프레임 디렉토리
            frames_output = database_adapter.get_output_path(video_info, 'frames',context['output_path'])
            self._ensure_directory(frames_output)
            
            # 이미 처리된 경우 건너뛰기
            if os.path.exists(frames_output) and len(os.listdir(frames_output)) > 0:
                self.logger.debug(f"Frames already extracted for {video_id}")
                video_info['frames_path'] = frames_output
                processed_videos.append(video_info)
                continue
            
            # 프레임 추출
            self.logger.debug(f"Extracting frames from {video_id}")
            success = self._extract_frames(video_path, frames_output)
            
            if success:
                video_info['frames_path'] = frames_output
                processed_videos.append(video_info)
            else:
                self.logger.warning(f"Failed to extract frames from {video_id}")
        
        # 업데이트된 비디오 파일 정보 저장
        context['video_files'] = processed_videos
        
        self._log_end(start_time, f"Extracted frames from {len(processed_videos)} videos")
        return context
    
    def _extract_frames(self, video_path: str, output_dir: str) -> bool:
        """
        비디오에서 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리 경로
            
        Returns:
            bool: 성공 여부
        """
        if self.use_ffmpeg:
            try:
                # FFmpeg 명령 구성
                nDataLoaderThread = 10
                command = [
                    'ffmpeg', '-y', 
                    '-i', video_path, 
                    '-qscale:v', '2', 
                    '-threads', str(nDataLoaderThread), 
                    '-r', str(self.frame_rate), 
                    '-f', 'image2', 
                    os.path.join(output_dir, '%06d.jpg'),
                    '-loglevel', 'panic'
                ]
                
                # FFmpeg 실행
                returncode, _, stderr = run_command(command, silent=True)
                
                if returncode != 0:
                    self.logger.warning(f"FFmpeg failed: {stderr}. Trying OpenCV.")
                    return self._extract_frames_opencv(video_path, output_dir)
                
                return True
            except Exception as e:
                self.logger.warning(f"Error with FFmpeg: {e}. Trying OpenCV.")
                return self._extract_frames_opencv(video_path, output_dir)
        else:
            return self._extract_frames_opencv(video_path, output_dir)
    
    def _extract_frames_opencv(self, video_path: str, output_dir: str) -> bool:
        """
        OpenCV를 사용하여 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return False
            
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = video.get(cv2.CAP_PROP_FPS)
            
            if original_fps <= 0:
                self.logger.warning(f"Invalid FPS: {original_fps}. Using default value.")
                original_fps = 30
            
            frame_interval = max(1, int(original_fps / self.frame_rate))
            
            count = 0
            frame_number = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                if count % frame_interval == 0:
                    frame_number += 1
                    output_path = os.path.join(output_dir, f'{frame_number:06d}.jpg')
                    cv2.imwrite(output_path, frame)
                
                count += 1
            
            video.release()
            
            # 최소 1개 이상의 프레임이 추출되었는지 확인
            if frame_number > 0:
                return True
            else:
                self.logger.warning(f"No frames extracted from {video_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error extracting frames with OpenCV: {e}")
            return False