#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from core.base import Processor, processor_registry
from core.utils import get_device

# EmoNet 모델 임포트
try:
    from emonet.emonet.models import EmoNet
    from emonet.emonet.data_augmentation import DataAugmentor
    from skimage import io
except ImportError:
    # 모듈 로드 실패 시 가짜 구현
    EmoNet = None
    DataAugmentor = None
    io = None

@processor_registry.register("VAFeatureExtractor")
class VAFeatureExtractor(Processor):
    """
    EmoNet을 사용하여 Valence-Arousal 특성을 추출하는 프로세서
    """
    
    def __init__(self, emonet_path: Optional[str] = None, n_expression: int = 8, **kwargs):
        """
        초기화
        
        Args:
            emonet_path: EmoNet 모델 경로
            n_expression: 표정 클래스 수
        """
        super().__init__(**kwargs)
        self.emonet_path = emonet_path
        self.n_expression = n_expression
        self.model = None
        self.logger.info("Initialized VAFeatureExtractor")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        프레임에서 VA 특성 추출
        
        Args:
            context: 처리 컨텍스트
                - 'video_files': 비디오 파일 정보 리스트
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                
        Returns:
            업데이트된 컨텍스트
        """
        start_time = self._log_start("Starting VA feature extraction")
        
        # 필요한 모듈 확인
        if EmoNet is None or DataAugmentor is None or io is None:
            self.logger.error("EmoNet, DataAugmentor, or io module is not installed")
            return context
        
        # 컨텍스트에서 필요한 정보 가져오기
        video_files = context.get('video_files', [])
        output_path = context.get('output_path')
        config = context.get('config')
        
        if not video_files:
            self.logger.warning("No video files found in context")
            return context
        
        # 출력 디렉토리 준비
        va_feature_root = os.path.join(output_path, 'VA_feature')
        self._ensure_directory(va_feature_root)
        
        # 장치 설정
        cuda_device = config.get('cuda_device', 0) if config.get('device') == 'cuda' else None
        device = get_device(cuda_device)
        
        # EmoNet 모델 초기화
        if self.model is None:
            try:
                if self.emonet_path and os.path.exists(self.emonet_path):
                    state_dict = torch.load(self.emonet_path, map_location='cpu')
                else:
                    # 경로가 없는 경우 기본 경로 사용
                    from pathlib import Path
                    state_dict_path = Path(__file__).parent.joinpath('emonet/pretrained', f'emonet_{self.n_expression}.pth')
                    if not os.path.exists(state_dict_path):
                        self.logger.error(f"EmoNet model not found at {state_dict_path}")
                        return context
                    state_dict = torch.load(str(state_dict_path), map_location='cpu')
                
                # module. 접두사 제거
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # 모델 초기화
                self.model = EmoNet(
                    extract=True, 
                    num_modules=2, 
                    n_expression=self.n_expression, 
                    n_reg=2, 
                    n_blocks=4, 
                    attention=True, 
                    temporal_smoothing=False
                ).to(device)
                
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.logger.info(f"Initialized EmoNet model on {device}")
            except Exception as e:
                self.logger.error(f"Failed to initialize EmoNet model: {e}")
                return context
        
        # 각 비디오에 대해 VA 특성 추출
        for video_info in tqdm(video_files, desc="Extracting VA features"):
            frames_path = video_info.get('frames_path')
            boxes_path = video_info.get('boxes_path')
            
            if not frames_path or not boxes_path:
                self.logger.warning(f"Frames or boxes not found for {video_info['id']}")
                continue
            
            if not os.path.exists(frames_path) or not os.path.exists(boxes_path):
                self.logger.warning(f"Frames or boxes directory not found for {video_info['id']}")
                continue
            
            # VA 특성 출력 디렉토리
            va_feature_output = os.path.join(va_feature_root, f"{video_info['id']}")
            self._ensure_directory(va_feature_output)
            
            # 이미 처리된 경우 건너뛰기
            va_feature_file = os.path.join(va_feature_output, 'VA_feature.npy')
            if os.path.exists(va_feature_file):
                self.logger.debug(f"VA features already extracted for {video_info['id']}")
                video_info['va_feature_path'] = va_feature_file
                continue
            
            # VA 특성 추출
            self.logger.debug(f"Extracting VA features for {video_info['id']}")
            success = self._extract_va_features(
                frames_path,
                boxes_path,
                va_feature_output,
                device
            )
            
            if success:
                video_info['va_feature_path'] = va_feature_file
                self.logger.debug(f"Extracted VA features for {video_info['id']}")
            else:
                self.logger.warning(f"Failed to extract VA features for {video_info['id']}")
        
        self._log_end(start_time, "VA feature extraction completed")
        return context
    
    def _extract_va_features(self, frames_path: str, boxes_path: str, output_dir: str, device: torch.device) -> bool:
        """
        프레임에서 VA 특성 추출
        
        Args:
            frames_path: 프레임 디렉토리 경로
            boxes_path: 박스 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            device: 계산 장치
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 데이터 증강기 초기화
            transform_image_shape = DataAugmentor(256, 256)
            transform_image = torch.nn.Sequential(
                torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
            
            # 박스 파일 목록 가져오기
            box_files = sorted([f for f in os.listdir(boxes_path) if f.endswith('.npy')])
            if not box_files:
                self.logger.warning(f"No box files found in {boxes_path}")
                return False
            
            # VA 특성 추출
            feature_list = []
            
            with torch.no_grad():
                for box_file in tqdm(box_files, desc="Processing frames", leave=False):
                    # 프레임 파일 경로
                    frame_name = os.path.splitext(box_file)[0] + '.jpg'
                    frame_path = os.path.join(frames_path, frame_name)
                    
                    if not os.path.exists(frame_path):
                        self.logger.debug(f"Frame not found: {frame_path}")
                        continue
                    
                    # 박스 로드
                    box = np.load(os.path.join(boxes_path, box_file))
                    
                    # 이미지 로드 및 전처리
                    image = io.imread(frame_path)
                    image, _ = transform_image_shape(image, bb=box)
                    image = np.ascontiguousarray(image)
                    
                    # 텐서 변환 및 정규화
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    image = transform_image(image).unsqueeze(0).to(device)
                    
                    # 특성 추출
                    output = self.model(image)
                    feature = output['feature'].cpu().numpy()
                    feature_list.append(feature)
            
            # 결과 저장
            if feature_list:
                feature_array = np.concatenate(feature_list, axis=0)
                output_file = os.path.join(output_dir, 'VA_feature.npy')
                np.save(output_file, feature_array)
                return True
            else:
                self.logger.warning("No features extracted")
                return False
            
        except Exception as e:
            self.logger.error(f"Error extracting VA features: {e}")
            return False