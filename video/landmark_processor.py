#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

from core.base import Processor, processor_registry
from core.utils import get_device

# face_alignment 모듈을 위한 설정
try:
    import face_alignment
except ImportError:
    face_alignment = None

@processor_registry.register("LandmarkProcessor")
class LandmarkProcessor(Processor):
    """
    얼굴 이미지에서 랜드마크를 추출하는 프로세서
    """
    
    def __init__(self, path_to_detector: Optional[str] = None, **kwargs):
        """
        초기화
        
        Args:
            path_to_detector: 감지기 모델 경로
        """
        super().__init__(**kwargs)
        self.path_to_detector = path_to_detector
        self.landmark_detector = None
        self.logger.info("Initialized LandmarkProcessor")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        얼굴 이미지에서 랜드마크 추출
        
        Args:
            context: 처리 컨텍스트
                - 'video_files': 비디오 파일 정보 리스트
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                
        Returns:
            업데이트된 컨텍스트
        """
        start_time = self._log_start("Starting landmark detection")
        
        # 필요한 모듈 확인
        if face_alignment is None:
            self.logger.error("face_alignment module is not installed")
            return context
        
        # 컨텍스트에서 필요한 정보 가져오기
        video_files = context.get('video_files', [])
        output_path = context.get('output_path')
        config = context.get('config')
                
        if not video_files:
            self.logger.warning("No video files found in context")
            return context
        
        # 설정 값 가져오기
        self.path_to_detector = config.get('video.landmark_detection.path_to_detector', self.path_to_detector)
        
        # 출력 디렉토리 준비
        landmarks_root = os.path.join(output_path, 'landmarks')
        boxes_root = os.path.join(output_path, 'boxes')
        
        self._ensure_directory(landmarks_root)
        self._ensure_directory(boxes_root)
        
        # 장치 설정
        cuda_device = config.get('cuda_device', 0) if config.get('device') == 'cuda' else None
        device = get_device(cuda_device)
        
        # 랜드마크 감지기 초기화
        if self.landmark_detector is None:
            try:
                detector_kwargs = {}
                if self.path_to_detector:
                    detector_kwargs['path_to_detector'] = self.path_to_detector
                
                self.landmark_detector = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    device=str(device),
                    flip_input=False,
                    face_detector_kwargs=detector_kwargs
                )
                self.logger.info(f"Initialized face alignment detector on {device}")
            except Exception as e:
                self.logger.error(f"Failed to initialize landmark detector: {e}")
                return context
        
        # 각 비디오에 대해 랜드마크 추출
        for video_info in tqdm(video_files, desc="Detecting landmarks"):
            faces_path = video_info.get('faces_path')
            if not faces_path or not os.path.exists(faces_path):
                self.logger.warning(f"Faces not found for {video_info['id']}")
                continue
            
            # 랜드마크 및 박스 출력 디렉토리
            landmarks_output = os.path.join(landmarks_root, f"{video_info['id']}")
            boxes_output = os.path.join(boxes_root, f"{video_info['id']}")
            log_dir = os.path.join(output_path, 'log', f"{video_info['id']}")
            
            self._ensure_directory(landmarks_output)
            self._ensure_directory(boxes_output)
            self._ensure_directory(log_dir)
            
            # 얼굴 이미지 목록 가져오기
            face_files = [f for f in os.listdir(faces_path) if f.endswith(('.jpg', '.png'))]
            if not face_files:
                self.logger.warning(f"No face images found in {faces_path}")
                continue
            
            # 이미 처리된 경우 건너뛰기
            if (os.path.exists(landmarks_output) and os.path.exists(boxes_output) and
                len(os.listdir(landmarks_output)) > 0 and len(os.listdir(boxes_output)) > 0):
                # 80% 이상 처리됨
                if (len(os.listdir(landmarks_output)) >= len(face_files) * 0.8 and
                    len(os.listdir(boxes_output)) >= len(face_files) * 0.8):
                    self.logger.debug(f"Landmarks already detected for {video_info['id']}")
                    video_info['landmarks_path'] = landmarks_output
                    video_info['boxes_path'] = boxes_output
                    continue
            
            # 랜드마크 추출
            self.logger.debug(f"Detecting landmarks in {video_info['id']}")
            landmarks_count = self._detect_landmarks(
                faces_path,
                landmarks_output,
                boxes_output,
                log_dir
            )
            
            if landmarks_count > 0:
                video_info['landmarks_path'] = landmarks_output
                video_info['boxes_path'] = boxes_output
                self.logger.debug(f"Detected landmarks for {landmarks_count} faces in {video_info['id']}")
            else:
                self.logger.warning(f"No landmarks detected in {video_info['id']}")
        
        self._log_end(start_time, "Landmark detection completed")
        return context
    
    def _detect_landmarks(self, faces_path: str, landmarks_output: str, 
                          boxes_output: str, log_dir: str) -> int:
        """
        얼굴 이미지에서 랜드마크 추출
        
        Args:
            faces_path: 얼굴 이미지 디렉토리 경로
            landmarks_output: 랜드마크 출력 디렉토리 경로
            boxes_output: 박스 출력 디렉토리 경로
            log_dir: 로그 디렉토리 경로
            
        Returns:
            int: 추출된 랜드마크 수
        """
        try:
            # face_alignment 라이브러리를 사용하여 디렉토리 내 모든 이미지에서 랜드마크 추출
            self.logger.info(f"Detecting landmarks from directory: {faces_path}")
            preds = self.landmark_detector.get_landmarks_from_directory(
                faces_path, 
                return_bboxes=True,
                show_progress_bar=True
            )
            
            if not preds:
                self.logger.warning(f"No landmarks detected in {faces_path}")
                return 0
            
            landmarks_count = 0
            
            # 결과 처리
            for image_file, (landmark, _, box) in preds.items():
                if not box:
                    with open(os.path.join(log_dir, 'log_not_box.txt'), 'a') as logger:
                        logger.write(os.path.abspath(image_file) + '\n')
                    continue
                
                # 첫 번째 얼굴만 사용
                landmark = np.array(landmark)[0]
                box = np.array(box)[0, :4]
                
                # 파일 이름 생성
                npy_file_name = os.path.splitext(os.path.basename(image_file))[0] + '.npy'
                landmark_path = os.path.join(landmarks_output, npy_file_name)
                box_path = os.path.join(boxes_output, npy_file_name)
                
                # 랜드마크와 박스 저장
                np.save(landmark_path, landmark)
                np.save(box_path, box)
                
                landmarks_count += 1
            
            return landmarks_count
            
        except Exception as e:
            self.logger.error(f"Error detecting landmarks: {e}")
            return 0