#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.base import DatabaseAdapter

def detect_database_type(root_path: str) -> str:
    """
    디렉토리 구조를 분석하여 데이터베이스 유형 자동 감지
    
    Args:
        root_path: 데이터베이스 루트 경로
        
    Returns:
        str: 감지된 데이터베이스 유형 ('lrs3', 'vox', 'standard')
    """
    logging.info(f"Detecting database type for: {root_path}")
    
    # 하위 폴더 체크
    potential_speaker_dirs = [d for d in os.listdir(root_path) 
                             if os.path.isdir(os.path.join(root_path, d))]
    
    # 폴더 내에 MP4가 있으면 LRS3 형식으로 취급
    if potential_speaker_dirs:
        found_mp4_in_subfolder = False
        
        for folder in potential_speaker_dirs:
            folder_path = os.path.join(root_path, folder)
            mp4_files = glob.glob(os.path.join(folder_path, "*.mp4"))
            
            if mp4_files:
                found_mp4_in_subfolder = True
                
                # LRS3 구조 확인: MP4와 대응하는 TXT가 있는지
                txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
                if txt_files:
                    mp4_base = os.path.splitext(os.path.basename(mp4_files[0]))[0]
                    if any(os.path.splitext(os.path.basename(txt))[0] == mp4_base for txt in txt_files):
                        logging.info("Detected LRS3-like database structure")
                        return 'lrs3'
        
        # 하위 폴더에 MP4는 있지만 대응하는 TXT가 없으면 표준 구조로 취급
        if found_mp4_in_subfolder:
            logging.info("Detected standard structure with videos in subfolders")
            return 'standard'
    
    # VoxCeleb 구조 확인: id/video_id/clip_id.mp4 형태 (3단계 구조)
    for potential_id in potential_speaker_dirs:
        id_dir = os.path.join(root_path, potential_id)
        video_dirs = [d for d in os.listdir(id_dir) if os.path.isdir(os.path.join(id_dir, d))]
        
        if video_dirs:
            sample_video_dir = os.path.join(id_dir, video_dirs[0])
            mp4_files = glob.glob(os.path.join(sample_video_dir, "*.mp4"))
            
            if mp4_files:
                logging.info("Detected VoxCeleb-like database structure")
                return 'vox'
    
    # 기본값은 표준 구조
    logging.info("Using standard database structure")
    return 'standard'

class StandardDatabaseAdapter(DatabaseAdapter):
    """
    기본 데이터베이스 구조 어댑터
    
    비디오/오디오 파일이 루트 디렉토리 또는 그 직접적인 하위 폴더에 있는 구조 처리
    """
    
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self.logger.info(f"Initialized StandardDatabaseAdapter with root: {root_path}")
    
    def get_video_files(self) -> List[Dict[str, str]]:
        """
        처리할 비디오 파일 목록 반환
        
        Returns:
            List[Dict[str, str]]: 비디오 파일 정보 목록
        """
        video_files = []
        
        # 루트 디렉토리에서 모든 mp4 파일 찾기
        mp4_pattern = os.path.join(self.root_path, "*.mp4")
        for video_path in glob.glob(mp4_pattern):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
            video_files.append({
                'path': video_path,
                'id': video_id,
                'folder': 'root',
                'index': video_id
            })
        
        # 루트 디렉토리의 모든 하위 폴더 검색
        subfolders = [d for d in os.listdir(self.root_path) 
                     if os.path.isdir(os.path.join(self.root_path, d))]
        
        for folder in subfolders:
            folder_path = os.path.join(self.root_path, folder)
            
            # 폴더 내 모든 mp4 파일 찾기
            mp4_pattern = os.path.join(folder_path, "*.mp4")
            for video_path in glob.glob(mp4_pattern):
                video_index = os.path.splitext(os.path.basename(video_path))[0]
                video_id = f"{folder}-{video_index}"
                
                video_files.append({
                    'path': video_path,
                    'id': video_id,
                    'folder': folder,
                    'index': video_index
                })
        
        self.logger.info(f"Found {len(video_files)} video files in standard structure")
        return video_files
    
    def get_audio_files(self) -> List[Dict[str, str]]:
        """
        처리할 오디오 파일 목록 반환
        
        Returns:
            List[Dict[str, str]]: 오디오 파일 정보 목록
        """
        audio_files = []
        corpus_dir = os.path.join(self.root_path, "corpus")
        
        # corpus 디렉토리가 존재하면 여기서 wav 파일 찾기
        if os.path.exists(corpus_dir) and os.path.isdir(corpus_dir):
            wav_pattern = os.path.join(corpus_dir, "*.wav")
            for audio_path in glob.glob(wav_pattern):
                audio_id = os.path.splitext(os.path.basename(audio_path))[0]
                
                # 대응하는 텍스트 파일 찾기
                transcript_path = os.path.join(corpus_dir, f"{audio_id}.txt")
                if not os.path.exists(transcript_path):
                    transcript_path = None
                
                audio_files.append({
                    'path': audio_path,
                    'id': audio_id,
                    'transcript_path': transcript_path
                })
            
            self.logger.info(f"Found {len(audio_files)} audio files in corpus directory")
            return audio_files
        
        # corpus 디렉토리가 없으면 비디오 파일에서 오디오 추출 필요
        self.logger.info("No corpus directory found, audio extraction from videos will be needed")
        
        # 비디오 파일 목록 가져와서 오디오 추출 필요 표시
        video_files = self.get_video_files()
        for video_info in video_files:
            video_path = video_info['path']
            video_id = video_info['id']
            folder = video_info['folder']
            index = video_info['index']
            
            # 대응하는 텍스트 파일 찾기 (폴더 내부 또는 루트에 있을 수 있음)
            transcript_path = None
            if folder != 'root':
                # 같은 폴더 내에서 찾기
                folder_transcript = os.path.join(self.root_path, folder, f"{index}.txt")
                if os.path.exists(folder_transcript):
                    transcript_path = folder_transcript
            
            if transcript_path is None:
                # 루트에서 찾기
                root_transcript = os.path.join(self.root_path, f"{index}.txt")
                if os.path.exists(root_transcript):
                    transcript_path = root_transcript
            
            audio_files.append({
                'path': video_path,  # 현재는 비디오 경로
                'id': video_id,
                'transcript_path': transcript_path,
                'needs_extraction': True,  # 오디오 추출 필요 표시
                'folder': folder,
                'index': index
            })
        
        return audio_files
        
    def get_output_path(self, file_info: Dict[str, str], category: str, output_root: str) -> str:
        folder = file_info.get('folder', 'root')
        index = file_info.get('index', file_info['id'])
        
        if folder == 'root':
            return os.path.join(output_root, category, index)
        else:
            if category in ['frames', 'faces', 'landmarks', 'mouth', 'boxes', 'VA_feature', 'mouth_emb']:
                return os.path.join(output_root, category, f"{folder}-{category}-{folder}-{index}")
            else:
                return os.path.join(output_root, category, f"{folder}-{index}")


class LRS3DatabaseAdapter(DatabaseAdapter):
    """
    LRS3 데이터베이스 구조 어댑터
    
    폴더 ID(스피커) -> 비디오 ID.mp4, 비디오 ID.txt 형태의 구조 처리
    """
    
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self.logger.info(f"Initialized LRS3DatabaseAdapter with root: {root_path}")
    
    def get_video_files(self) -> List[Dict[str, str]]:
        """
        처리할 비디오 파일 목록 반환
        
        Returns:
            List[Dict[str, str]]: 비디오 파일 정보 목록
        """
        video_files = []
        
        # 모든 스피커 폴더 탐색
        speaker_folders = [d for d in os.listdir(self.root_path) 
                           if os.path.isdir(os.path.join(self.root_path, d))]
        
        for folder in speaker_folders:
            folder_path = os.path.join(self.root_path, folder)
            
            # 폴더 내 모든 mp4 파일 찾기
            mp4_pattern = os.path.join(folder_path, "*.mp4")
            for video_path in glob.glob(mp4_pattern):
                video_index = os.path.splitext(os.path.basename(video_path))[0]
                
                # 비디오 폴더 이름을 ID에 포함
                root_folder_name = os.path.basename(self.root_path)
                video_id = f"{folder}_{video_index}"
                
                video_files.append({
                    'path': video_path,
                    'id': video_id,
                    'folder': folder,
                    'index': video_index
                })
        
        self.logger.info(f"Found {len(video_files)} video files in LRS3 structure")
        return video_files
    
    def get_audio_files(self) -> List[Dict[str, str]]:
        """
        처리할 오디오 파일 목록 반환
        
        먼저 corpus 디렉토리에서 wav 파일을 찾고, 없으면 각 폴더에서 mp4를 기반으로 오디오 추출이 필요함을 표시
        
        Returns:
            List[Dict[str, str]]: 오디오 파일 정보 목록
        """
        audio_files = []
        corpus_dir = os.path.join(self.root_path, "corpus")
        
        # corpus 디렉토리가 존재하면 여기서 wav 파일 찾기
        if os.path.exists(corpus_dir) and os.path.isdir(corpus_dir):
            wav_pattern = os.path.join(corpus_dir, "*.wav")
            for audio_path in glob.glob(wav_pattern):
                audio_id = os.path.splitext(os.path.basename(audio_path))[0]
                
                # 대응하는 텍스트 파일 찾기
                transcript_path = os.path.join(corpus_dir, f"{audio_id}.txt")
                if not os.path.exists(transcript_path):
                    transcript_path = None
                
                audio_files.append({
                    'path': audio_path,
                    'id': audio_id,
                    'transcript_path': transcript_path
                })
            
            self.logger.info(f"Found {len(audio_files)} audio files in corpus directory")
            return audio_files
        
        # corpus 디렉토리가 없으면 비디오 파일에서 오디오 추출 필요
        self.logger.info("No corpus directory found, audio extraction from videos will be needed")
        
        # 비디오 파일 목록 가져와서 오디오 추출 필요 표시
        video_files = self.get_video_files()
        for video_info in video_files:
            video_path = video_info['path']
            video_id = video_info['id']
            folder = video_info['folder']
            index = video_info['index']
            
            # 대응하는 텍스트 파일 찾기
            transcript_path = os.path.join(self.root_path, folder, f"{index}.txt")
            if not os.path.exists(transcript_path):
                transcript_path = None
            
            audio_files.append({
                'path': video_path,  # 현재는 비디오 경로
                'id': video_id,
                'transcript_path': transcript_path,
                'needs_extraction': True,  # 오디오 추출 필요 표시
                'folder': folder,
                'index': index
            })
        
        return audio_files
    
    def get_output_path(self, file_info: Dict[str, str], category: str, output_root: str) -> str:
        folder = file_info.get('folder', 'root')
        index = file_info.get('index', file_info['id'])
        
        if folder == 'root':
            return os.path.join(output_root, category, index)
        else:
            if category in ['frames', 'faces', 'landmarks', 'mouth', 'boxes', 'VA_feature', 'mouth_emb']:
                return os.path.join(output_root, category, f"{folder}-{category}-{folder}-{index}")
            else:
                return os.path.join(output_root, category, f"{folder}-{index}")


class VoxDatabaseAdapter(DatabaseAdapter):
    """
    VoxCeleb 데이터베이스 구조 어댑터
    
    id/video_id/clip_id.mp4 형태의 구조 처리
    """
    
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self.logger.info(f"Initialized VoxDatabaseAdapter with root: {root_path}")
    
    def get_video_files(self) -> List[Dict[str, str]]:
        """
        처리할 비디오 파일 목록 반환
        
        Returns:
            List[Dict[str, str]]: 비디오 파일 정보 목록
        """
        video_files = []
        
        # 모든 ID 폴더 탐색
        id_folders = [d for d in os.listdir(self.root_path) 
                      if os.path.isdir(os.path.join(self.root_path, d))]
        
        for id_folder in id_folders:
            id_path = os.path.join(self.root_path, id_folder)
            
            # ID 폴더 내 모든 비디오 ID 폴더 탐색
            video_folders = [d for d in os.listdir(id_path) 
                             if os.path.isdir(os.path.join(id_path, d))]
            
            for video_folder in video_folders:
                video_path = os.path.join(id_path, video_folder)
                
                # 비디오 폴더 내 모든 mp4 파일 찾기
                mp4_pattern = os.path.join(video_path, "*.mp4")
                for video_file in glob.glob(mp4_pattern):
                    clip_id = os.path.splitext(os.path.basename(video_file))[0]
                    video_id = f"{id_folder}_{video_folder}_{clip_id}"
                    
                    video_files.append({
                        'path': video_file,
                        'id': video_id,
                        'folder': f"{id_folder}/{video_folder}",
                        'index': clip_id
                    })
        
        self.logger.info(f"Found {len(video_files)} video files in VoxCeleb structure")
        return video_files
    
    def get_audio_files(self) -> List[Dict[str, str]]:
        """
        처리할 오디오 파일 목록 반환
        
        VoxCeleb은 일반적으로 텍스트 대본이 없으므로 오디오만 추출
        
        Returns:
            List[Dict[str, str]]: 오디오 파일 정보 목록
        """
        audio_files = []
        corpus_dir = os.path.join(self.root_path, "corpus")
        
        # corpus 디렉토리가 존재하면 여기서 wav 파일 찾기
        if os.path.exists(corpus_dir) and os.path.isdir(corpus_dir):
            wav_pattern = os.path.join(corpus_dir, "*.wav")
            for audio_path in glob.glob(wav_pattern):
                audio_id = os.path.splitext(os.path.basename(audio_path))[0]
                
                audio_files.append({
                    'path': audio_path,
                    'id': audio_id,
                    'transcript_path': None  # VoxCeleb은 대본이 없음
                })
            
            self.logger.info(f"Found {len(audio_files)} audio files in corpus directory")
            return audio_files
        
        # corpus 디렉토리가 없으면 비디오 파일에서 오디오 추출 필요
        self.logger.info("No corpus directory found, audio extraction from videos will be needed")
        
        # 비디오 파일 목록 가져와서 오디오 추출 필요 표시
        video_files = self.get_video_files()
        for video_info in video_files:
            video_path = video_info['path']
            video_id = video_info['id']
            folder = video_info['folder']
            index = video_info['index']
            
            audio_files.append({
                'path': video_path,  # 현재는 비디오 경로
                'id': video_id,
                'transcript_path': None,  # VoxCeleb은 대본이 없음
                'needs_extraction': True,  # 오디오 추출 필요 표시
                'folder': folder,
                'index': index
            })
        
        return audio_files
    
    def get_output_path(self, file_info: Dict[str, str], category: str, output_root: str) -> str:
        folder = file_info.get('folder', 'root')
        index = file_info.get('index', file_info['id'])
        
        if folder == 'root':
            return os.path.join(output_root, category, index)
        else:
            if category in ['frames', 'faces', 'landmarks', 'mouth', 'boxes', 'VA_feature', 'mouth_emb']:
                return os.path.join(output_root, category, f"{folder}-{category}-{folder}-{index}")
            else:
                return os.path.join(output_root, category, f"{folder}-{index}")