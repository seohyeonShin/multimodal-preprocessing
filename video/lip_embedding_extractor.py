#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from core.base import Processor, processor_registry
from core.utils import get_device

# Lipreading 모델 임포트
try:
    from lipreading.model import Lipreading
    from lipreading.utils import load_json, calculateNorm2, load_model
    from lipreading.dataloaders import get_preprocessing_pipelines
except ImportError:
    # 모듈 로드 실패 시 가짜 구현
    Lipreading = None
    load_json = None
    calculateNorm2 = None
    load_model = None
    get_preprocessing_pipelines = None

@processor_registry.register("LipEmbeddingExtractor")
class LipEmbeddingExtractor(Processor):
    """
    립리딩 모델을 사용하여 입 영역 임베딩을 추출하는 프로세서
    """
    
    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None, **kwargs):
        """
        초기화
        
        Args:
            config_path: 립리딩 모델 설정 파일 경로
            model_path: 립리딩 모델 파일 경로
        """
        super().__init__(**kwargs)
        self.config_path = config_path
        self.model_path = model_path
        self.model = None
        self.logger.info("Initialized LipEmbeddingExtractor")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        입 영역 이미지에서 임베딩 추출
        
        Args:
            context: 처리 컨텍스트
                - 'video_files': 비디오 파일 정보 리스트
                - 'output_path': 출력 디렉토리 경로
                - 'config': 설정 객체
                
        Returns:
            업데이트된 컨텍스트
        """
        start_time = self._log_start("Starting lip embedding extraction")
        
        # 필요한 모듈 확인
        if Lipreading is None or load_json is None or calculateNorm2 is None or load_model is None:
            self.logger.error("Lipreading modules are not installed")
            return context
        
        # 컨텍스트에서 필요한 정보 가져오기
        video_files = context.get('video_files', [])
        output_path = context.get('output_path')
        config = context.get('config')
        
        if not video_files:
            self.logger.warning("No video files found in context")
            return context
        
        # 설정 및 모델 경로 가져오기
        if not self.config_path:
            self.config_path = config.get('video.lip_embedding.config_path', 'configs/lrw_resnet18_mstcn.json')
        
        if not self.model_path:
            self.model_path = config.get('video.lip_embedding.model_path', 'model/lipreading/lrw_resnet18_mstcn_video.pth')
        
        # 설정 파일 확인
        if not os.path.exists(self.config_path):
            self.logger.error(f"Lipreading config file not found: {self.config_path}")
            return context
        
        # 모델 파일 확인
        if not os.path.exists(self.model_path):
            self.logger.error(f"Lipreading model file not found: {self.model_path}")
            return context
        
        # 출력 디렉토리 준비
        mouth_emb_root = os.path.join(output_path, 'mouth_emb')
        self._ensure_directory(mouth_emb_root)
        
        # 장치 설정
        cuda_device = config.get('cuda_device', 0) if config.get('device') == 'cuda' else None
        device = get_device(cuda_device)
        
        # 모델 초기화
        if self.model is None:
            try:
                # 설정 로드
                args_loaded = load_json(self.config_path)
                
                # TCN 옵션 설정
                tcn_options = {
                    'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                }
                
                # 모델 초기화
                self.model = Lipreading(
                    modality='video',
                    num_classes=500,
                    tcn_options=tcn_options,
                    densetcn_options={},
                    backbone_type=args_loaded['backbone_type'],
                    relu_type=args_loaded['relu_type'],
                    width_mult=args_loaded['width_mult'],
                    use_boundary=args_loaded.get("use_boundary", False),
                    extract_feats=True
                ).to(device)
                
                calculateNorm2(self.model)
                self.model = load_model(self.model_path, self.model, allow_size_mismatch=False)
                self.model.eval()
                self.logger.info(f"Initialized Lipreading model on {device}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Lipreading model: {e}")
                return context
        
        # 각 비디오에 대해 임베딩 추출
        for video_info in tqdm(video_files, desc="Extracting lip embeddings"):
            mouth_npz_path = video_info.get('mouth_npz_path')
            if not mouth_npz_path:
                # NPZ 경로가 없으면 구성해보기
                mouth_path = video_info.get('mouth_path')
                if mouth_path:
                    mouth_npz_path = os.path.join(os.path.dirname(mouth_path), f"{os.path.basename(mouth_path)}.npz")
            
            if not mouth_npz_path or not os.path.exists(mouth_npz_path):
                self.logger.warning(f"Mouth NPZ file not found for {video_info['id']}")
                continue
            
            # 임베딩 출력 디렉토리
            mouth_emb_output = os.path.join(mouth_emb_root, f"{video_info['id']}")
            self._ensure_directory(mouth_emb_output)
            
            # 이미 처리된 경우 건너뛰기
            mouth_emb_file = os.path.join(mouth_emb_output, 'mouth_emb.npy')
            if os.path.exists(mouth_emb_file):
                self.logger.debug(f"Lip embeddings already extracted for {video_info['id']}")
                video_info['mouth_emb_path'] = mouth_emb_file
                continue
            
            # 임베딩 추출
            self.logger.debug(f"Extracting lip embeddings for {video_info['id']}")
            success = self._extract_lip_embeddings(
                mouth_npz_path,
                mouth_emb_file
            )
            
            if success:
                video_info['mouth_emb_path'] = mouth_emb_file
                self.logger.debug(f"Extracted lip embeddings for {video_info['id']}")
            else:
                self.logger.warning(f"Failed to extract lip embeddings for {video_info['id']}")
        
        self._log_end(start_time, "Lip embedding extraction completed")
        return context
    
    def _extract_lip_embeddings(self, npz_path: str, output_file: str) -> bool:
        """
        입 영역 NPZ 파일에서 임베딩 추출
        
        Args:
            npz_path: 입 영역 NPZ 파일 경로
            output_file: 출력 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 전처리 파이프라인 가져오기
            preprocessing_func = get_preprocessing_pipelines('video')['test']
            
            # NPZ 데이터 로드 및 전처리
            data = preprocessing_func(np.load(npz_path)['data'])  # data: TxHxW
            
            # 텐서 변환
            data_tensor = torch.FloatTensor(data)
            data_var = data_tensor.unsqueeze(0).unsqueeze(0).to(next(self.model.parameters()).device)
            
            # 임베딩 추출
            with torch.no_grad():
                output = self.model(data_var, lengths=[data.shape[0]])
            
            if output is None:
                self.logger.error(f"Model returned None for {npz_path}")
                return False
            
            # 결과 저장
            np.save(output_file, output.cpu().detach().numpy())
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            self.logger.error(f"Error extracting lip embeddings: {e}")
            return False