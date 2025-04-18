#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
from collections import deque
from skimage import transform as tf
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

from core.base import Processor, processor_registry

@processor_registry.register("MouthExtractor")
class MouthExtractor(Processor):
    """
    랜드마크를 기반으로 입 영역을 추출하는 프로세서
    """
    
    def __init__(self, mouth_width: int = 96, mouth_height: int = 96, 
                 start_idx: int = 48, stop_idx: int = 68, window_margin: int = 12,
                 mean_face_path: Optional[str] = None, **kwargs):
        """
        초기화
        
        Args:
            mouth_width: 입 영역 너비
            mouth_height: 입 영역 높이
            start_idx: 입 랜드마크 시작 인덱스
            stop_idx: 입 랜드마크 종료 인덱스
            window_margin: 시간적 스무딩을 위한 윈도우 마진
            mean_face_path: 평균 얼굴 랜드마크 파일 경로
        """
        super().__init__(**kwargs)
        self.mouth_width = mouth_width
        self.mouth_height = mouth_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.mean_face_path = mean_face_path
        self.logger.info("Initialized MouthExtractor")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        랜드마크를 기반으로 입 영역 추출
        
        Args:
            context: 처리 컨텍스트
                - 'video_files': 비디오 파일 정보 리스트
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                
        Returns:
            업데이트된 컨텍스트
        """
        start_time = self._log_start("Starting mouth region extraction")
        
        # 컨텍스트에서 필요한 정보 가져오기
        video_files = context.get('video_files', [])
        output_path = context.get('output_path')
        config = context.get('config')
        
        if not video_files:
            self.logger.warning("No video files found in context")
            return context
        
        # 설정 값 가져오기
        self.mouth_width = config.get('video.mouth.width', self.mouth_width)
        self.mouth_height = config.get('video.mouth.height', self.mouth_height)
        self.start_idx = config.get('video.mouth.start_idx', self.start_idx)
        self.stop_idx = config.get('video.mouth.stop_idx', self.stop_idx)
        self.window_margin = config.get('video.mouth.window_margin', self.window_margin)
        self.mean_face_path = config.get('video.mouth.mean_face_path', self.mean_face_path)
        
        # 기본 평균 얼굴 랜드마크 파일
        if not self.mean_face_path or not os.path.exists(self.mean_face_path):
            self.mean_face_path = '20words_mean_face.npy'
            if not os.path.exists(self.mean_face_path):
                # 기본 평균 얼굴 랜드마크 생성
                self.logger.warning(f"Mean face file not found: {self.mean_face_path}")
                self.logger.info("Creating default mean face landmarks")
                mean_face = self._create_default_mean_face()
                np.save(self.mean_face_path, mean_face)
        
        # 평균 얼굴 랜드마크 로드
        try:
            mean_face_landmarks = np.load(self.mean_face_path)
            self.logger.info(f"Loaded mean face landmarks from {self.mean_face_path}")
        except Exception as e:
            self.logger.error(f"Error loading mean face landmarks: {e}")
            self.logger.info("Creating default mean face landmarks")
            mean_face_landmarks = self._create_default_mean_face()
        
        # 출력 디렉토리 준비
        mouth_root = os.path.join(output_path, 'mouth')
        self._ensure_directory(mouth_root)
        
        # 각 비디오에 대해 입 영역 추출
        for video_info in tqdm(video_files, desc="Extracting mouth regions"):
            faces_path = video_info.get('faces_path')
            landmarks_path = video_info.get('landmarks_path')
            
            if not faces_path or not landmarks_path:
                self.logger.warning(f"Faces or landmarks not found for {video_info['id']}")
                continue
            
            if not os.path.exists(faces_path) or not os.path.exists(landmarks_path):
                self.logger.warning(f"Faces or landmarks directory not found for {video_info['id']}")
                continue
            
            # 입 영역 출력 디렉토리
            mouth_output = os.path.join(mouth_root, f"{video_info['id']}")
            self._ensure_directory(mouth_output)
            
            # 이미 처리된 경우 건너뛰기
            npz_path = os.path.join(os.path.dirname(mouth_output), f"{os.path.basename(mouth_output)}.npz")
            if os.path.exists(npz_path):
                self.logger.debug(f"Mouth regions already extracted for {video_info['id']}")
                video_info['mouth_path'] = mouth_output
                video_info['mouth_npz_path'] = npz_path
                continue
            
            # 입 영역 추출
            self.logger.debug(f"Extracting mouth regions for {video_info['id']}")
            success = self._extract_mouth_regions(
                faces_path,
                landmarks_path,
                mouth_output,
                mean_face_landmarks
            )
            
            if success:
                video_info['mouth_path'] = mouth_output
                video_info['mouth_npz_path'] = npz_path
                self.logger.debug(f"Extracted mouth regions for {video_info['id']}")
            else:
                self.logger.warning(f"Failed to extract mouth regions for {video_info['id']}")
        
        self._log_end(start_time, "Mouth region extraction completed")
        return context
    
    def _extract_mouth_regions(self, faces_path: str, landmarks_path: str, 
                              output_dir: str, mean_face_landmarks: np.ndarray) -> bool:
        """
        얼굴 이미지와 랜드마크를 사용하여 입 영역 추출
        
        Args:
            faces_path: 얼굴 이미지 디렉토리 경로
            landmarks_path: 랜드마크 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            mean_face_landmarks: 평균 얼굴 랜드마크
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 얼굴 이미지 파일 목록
            frame_files = sorted([f for f in os.listdir(faces_path) if f.endswith(('.jpg', '.png'))])
            if not frame_files:
                self.logger.warning(f"No face images found in {faces_path}")
                return False
            
            # 입 영역 추출
            self._crop_mouth_images(
                faces_path,
                landmarks_path,
                output_dir,
                mean_face_landmarks,
                crop_width=self.mouth_width // 2,
                crop_height=self.mouth_height // 2,
                start_idx=self.start_idx,
                stop_idx=self.stop_idx,
                window_margin=self.window_margin
            )
            
            # NPZ 파일 생성
            mouth_images = sorted([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))])
            if not mouth_images:
                self.logger.warning(f"No mouth images extracted to {output_dir}")
                return False
            
            sequence = []
            for img_file in mouth_images:
                img_path = os.path.join(output_dir, img_file)
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                sequence.append(gray_img)
            
            # 디렉토리 이름으로 NPZ 파일 저장
            npz_path = os.path.join(os.path.dirname(output_dir), f"{os.path.basename(output_dir)}.npz")
            np.savez_compressed(npz_path, data=np.array(sequence))
            
            return True
        except Exception as e:
            self.logger.error(f"Error extracting mouth regions: {e}")
            return False
    
    def _crop_mouth_images(self, video_path: str, landmarks_dir: str, target_dir: str,
                          mean_face_landmarks: np.ndarray, crop_width: int, crop_height: int,
                          start_idx: int, stop_idx: int, window_margin: int) -> None:
        """
        입 영역 추출 및 저장
        """
        STD_SIZE = (256, 256)
        STABLE_POINTS = [33, 36, 39, 42, 45]  # 코와 눈의 안정적인 랜드마크 포인트
        
        frame_names = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        
        q_frames, q_landmarks, q_name = deque(), deque(), deque()
        
        for frame_name in tqdm(frame_names, desc="Processing mouth regions", leave=False):
            landmark_path = os.path.join(landmarks_dir, f'{os.path.splitext(frame_name)[0]}.npy')
            if not os.path.exists(landmark_path):
                continue
            
            landmarks = np.load(landmark_path)
            
            with Image.open(os.path.join(video_path, frame_name)) as pil_img:
                img = np.array(pil_img)
            
            # 큐에 요소 추가
            q_frames.append(img)
            q_landmarks.append(landmarks)
            q_name.append(frame_name)
            
            if len(q_frames) == window_margin:  # 큐가 충분히 차면 처리 시작
                # 시간적 스무딩을 위해 랜드마크 평균 계산
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                
                # 큐에서 현재 프레임 데이터 가져오기
                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frames.popleft()
                cur_name = q_name.popleft()
                
                # 얼굴 정렬 및 변환 행렬 계산
                trans_frame, trans = self._warp_img(
                    smoothed_landmarks[STABLE_POINTS, :],
                    mean_face_landmarks[STABLE_POINTS, :],
                    cur_frame,
                    STD_SIZE
                )
                
                # 랜드마크에 변환 행렬 적용
                trans_landmarks = trans(cur_landmarks)
                
                # 입 영역 추출
                try:
                    cropped_frame = self._cut_patch(
                        trans_frame,
                        trans_landmarks[start_idx:stop_idx],
                        crop_height,
                        crop_width
                    )
                    
                    # 이미지 저장
                    target_path = os.path.join(target_dir, cur_name)
                    Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)
                except Exception as e:
                    self.logger.error(f"Error cropping mouth region: {e}")
        
        # 남은 프레임 처리
        while q_frames:
            cur_frame = q_frames.popleft()
            cur_name = q_name.popleft()
            cur_landmarks = q_landmarks.popleft()
            
            try:
                if 'trans' in locals():  # trans가 정의된 경우에만 처리
                    trans_frame = self._apply_transform(trans, cur_frame, STD_SIZE)
                    trans_landmarks = trans(cur_landmarks)
                    
                    cropped_frame = self._cut_patch(
                        trans_frame,
                        trans_landmarks[start_idx:stop_idx],
                        crop_height,
                        crop_width
                    )
                    
                    target_path = os.path.join(target_dir, cur_name)
                    Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)
            except Exception as e:
                self.logger.error(f"Error processing remaining frame {cur_name}: {e}")
    
    def _cut_patch(self, img: np.ndarray, landmarks: np.ndarray, 
                  height: int, width: int, threshold: int = 5) -> np.ndarray:
        """
        랜드마크 중심으로 패치 추출
        """
        center_x, center_y = np.mean(landmarks, axis=0)
        
        # 경계 확인 및 조정
        if center_y - height < 0:
            center_y = height
        if int(center_y) - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')
        if center_y + height > img.shape[0]:
            center_y = img.shape[0] - height
        if center_y + height > img.shape[0] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[1]:
            center_x = img.shape[1] - width
        if center_x + width > img.shape[1] + threshold:
            raise Exception('too much bias in width')
        
        # 패치 추출
        img_cropped = np.copy(
            img[
                int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                int(round(center_x) - round(width)): int(round(center_x) + round(width)),
            ]
        )
        return img_cropped
    
    def _apply_transform(self, transform, img: np.ndarray, std_size: Tuple[int, int]) -> np.ndarray:
        """
        이미지에 변환 행렬 적용
        """
        warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
        warped = warped * 255
        warped = warped.astype('uint8')
        return warped
    
    def _warp_img(self, src: np.ndarray, dst: np.ndarray, img: np.ndarray, 
                 std_size: Tuple[int, int]) -> Tuple[np.ndarray, Any]:
        """
        랜드마크 맞춤을 위한 이미지 와핑
        """
        tform = tf.estimate_transform('similarity', src, dst)  # 변환 행렬 계산
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # 이미지 와핑
        warped = warped * 255
        warped = warped.astype('uint8')
        return warped, tform
    
    def _create_default_mean_face(self) -> np.ndarray:
        """
        기본 평균 얼굴 랜드마크 생성
        """
        # 68개 랜드마크 포인트를 가진 기본 평균 얼굴
        # 실제 애플리케이션에서는 더 정확한 값을 사용해야 함
        mean_face = np.zeros((68, 2))
        
        # 얼굴 윤곽
        mean_face[0:17] = np.array([
            [0.00, 0.00], [0.10, 0.00], [0.20, 0.00], [0.30, 0.00], [0.40, 0.00], 
            [0.50, 0.10], [0.60, 0.10], [0.70, 0.10], [0.80, 0.20], [0.90, 0.30], 
            [0.90, 0.40], [0.90, 0.50], [0.80, 0.60], [0.70, 0.70], [0.60, 0.70], 
            [0.50, 0.70], [0.40, 0.70]
        ])
        
        # 눈썹
        mean_face[17:27] = np.array([
            [0.20, 0.30], [0.30, 0.30], [0.40, 0.30], [0.50, 0.30], [0.60, 0.30],
            [0.70, 0.30], [0.80, 0.30], [0.90, 0.30], [0.80, 0.30], [0.70, 0.30]
        ])
        
        # 코
        mean_face[27:36] = np.array([
            [0.50, 0.40], [0.50, 0.50], [0.50, 0.60], [0.40, 0.60], [0.45, 0.60],
            [0.50, 0.60], [0.55, 0.60], [0.60, 0.60], [0.50, 0.60]
        ])
        
        # 눈
        mean_face[36:48] = np.array([
            [0.30, 0.40], [0.35, 0.40], [0.40, 0.40], [0.45, 0.40], [0.40, 0.40], [0.35, 0.40],
            [0.55, 0.40], [0.60, 0.40], [0.65, 0.40], [0.70, 0.40], [0.65, 0.40], [0.60, 0.40]
        ])
        
        # 입
        mean_face[48:68] = np.array([
            [0.35, 0.75], [0.40, 0.75], [0.45, 0.75], [0.50, 0.75], [0.55, 0.75], [0.60, 0.75], [0.65, 0.75],
            [0.60, 0.80], [0.55, 0.80], [0.50, 0.80], [0.45, 0.80], [0.40, 0.80],
            [0.40, 0.75], [0.45, 0.75], [0.50, 0.75], [0.55, 0.75], [0.60, 0.75],
            [0.55, 0.75], [0.50, 0.75], [0.45, 0.75]
        ])
        
        # 0~1 범위에서 0~255 범위로 변환
        mean_face = mean_face * 255
        
        return mean_face