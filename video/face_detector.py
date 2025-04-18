#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

from core.base import Processor, processor_registry
from core.utils import get_device

# S3FD 모델을 위한 설정 (실제 모델은 따로 로드해야 함)
# try:
from video.model.faceDetector.s3fd import S3FD
# except ImportError:
#     # 모듈 로드 실패 시 가짜 구현
#     class S3FD:
#         def __init__(self, device='cuda'):
#             self.device = device
            
#         def detect_faces(self, image, conf_th=0.9, scales=[0.25]):
#             # 임시 구현 - 실제 모델이 없을 때
#             h, w = image.shape[:2]
#             return [[int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]]

@processor_registry.register("FaceDetector")
class FaceDetector(Processor):
    """
    프레임에서 얼굴을 감지하는 프로세서
    """
    
    def __init__(self, confidence_threshold: float = 0.9, scales: List[float] = [0.25], **kwargs):
        """
        초기화
        
        Args:
            confidence_threshold: 얼굴 감지 신뢰도 임계값
            scales: 감지 스케일 리스트
        """
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.scales = scales
        self.detector = None
        self.logger.info(f"Initialized FaceDetector (confidence_threshold={confidence_threshold})")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        프레임에서 얼굴 감지 및 추출
        
        Args:
            context: 처리 컨텍스트
                - 'video_files': 비디오 파일 정보 리스트
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                
        Returns:
            업데이트된 컨텍스트
        """
        start_time = self._log_start("Starting face detection")
        
        # 컨텍스트에서 필요한 정보 가져오기
        video_files = context.get('video_files', [])
        output_path = context.get('output_path')
        config = context.get('config')
        
        if not video_files:
            self.logger.warning("No video files found in context")
            return context
        
        # 설정 값 가져오기
        self.confidence_threshold = config.get('video.face_detection.confidence_threshold', self.confidence_threshold)
        self.scales = config.get('video.face_detection.scales', self.scales)
        
        # 출력 디렉토리 준비
        faces_root = os.path.join(output_path, 'faces')
        self._ensure_directory(faces_root)
        
        # 장치 설정 및 감지기 초기화
        cuda_device = config.get('cuda_device', 0) if config.get('device') == 'cuda' else None
        device = get_device(cuda_device)
        
        # 감지기 초기화
        if self.detector is None:
            try:
                self.detector = S3FD(device=device)
                self.logger.info(f"Initialized S3FD face detector on {device}")
            except Exception as e:
                self.logger.error(f"Failed to initialize face detector: {e}")
                return context
        
        # 각 비디오에 대해 얼굴 감지
        for video_info in tqdm(video_files, desc="Detecting faces"):
            frames_path = video_info.get('frames_path')
            if not frames_path or not os.path.exists(frames_path):
                self.logger.warning(f"Frames not found for {video_info['id']}")
                continue
            
            # 얼굴 출력 디렉토리
            faces_output = os.path.join(faces_root, f"{video_info['id']}")
            self._ensure_directory(faces_output)
            
            # 프레임 목록 가져오기
            frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
            if not frame_files:
                self.logger.warning(f"No frames found in {frames_path}")
                continue
            
            # 이미 처리된 경우 건너뛰기
            if os.path.exists(faces_output) and len(os.listdir(faces_output)) > 0:
                if len(os.listdir(faces_output)) >= len(frame_files) * 0.8:  # 80% 이상 처리됨
                    self.logger.debug(f"Faces already detected for {video_info['id']}")
                    video_info['faces_path'] = faces_output
                    continue
            
            # 프레임에서 얼굴 감지
            self.logger.debug(f"Detecting faces in {video_info['id']}")
            face_count = self._detect_faces_in_frames(
                frames_path, 
                faces_output, 
                frame_files, 
                direction=config.get('video.face_detection.direction', 'right')
            )
            
            if face_count > 0:
                video_info['faces_path'] = faces_output
                self.logger.debug(f"Detected {face_count} faces in {video_info['id']}")
            else:
                self.logger.warning(f"No faces detected in {video_info['id']}")
        
        self._log_end(start_time, "Face detection completed")
        return context
    
    def _detect_faces_in_frames(self, frames_path: str, output_dir: str, 
                               frame_files: List[str], direction: str = 'all') -> int:
        """
        프레임 폴더에서 얼굴 감지
        
        Args:
            frames_path: 프레임 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            frame_files: 프레임 파일 목록
            direction: 얼굴 감지 방향 ('right', 'left', 'all')
            
        Returns:
            int: 감지된 얼굴 수
        """
        face_count = 0
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_path, frame_file)
            output_file = os.path.join(output_dir, frame_file)
            
            # 이미 처리된 경우 건너뛰기
            if os.path.exists(output_file):
                face_count += 1
                continue
            
            try:
                # 이미지 로드
                image = cv2.imread(frame_path)
                if image is None:
                    self.logger.warning(f"Could not read image: {frame_path}")
                    continue
                
                # BGR -> RGB 변환
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 이미지 중심 계산
                image_width = image_rgb.shape[1]
                image_center_x = image_width / 2
                
                # 얼굴 감지
                bboxes = self.detector.detect_faces(
                    image_rgb, 
                    conf_th=self.confidence_threshold, 
                    scales=self.scales
                )
                
                face_detected = False
                for bbox in bboxes:
                    if bbox is not None:
                        bbox_center_x = (bbox[0] + bbox[2]) / 2
                        
                        # # 방향에 따라 얼굴 선택
                        # if direction == 'right' and bbox_center_x <= image_center_x:
                        #     continue
                        # elif direction == 'left' and bbox_center_x >= image_center_x:
                        #     continue
                        
                        # 얼굴 영역 추출
                        face_image = Image.fromarray(image_rgb)
                        face_image = face_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        
                        if face_image is not None:
                            face_image.save(output_file)
                            face_detected = True
                            face_count += 1
                            break
                
                if not face_detected:
                    self.logger.debug(f"No face detected in {frame_file}")
                    
            except Exception as e:
                self.logger.error(f"Error detecting face in {frame_file}: {e}")
        
        return face_count