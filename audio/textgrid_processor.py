#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path
from tqdm import tqdm
import traceback
import shutil

from core.base import Processor, processor_registry
from core.utils import run_command

@processor_registry.register("TextGridProcessor")
class TextGridProcessor(Processor):
    """
    Montreal Forced Aligner를 사용하여 오디오와 텍스트를 정렬하고 TextGrid 파일 생성
    전체 corpus를 한 번에 처리하도록 개선
    """
    
    def __init__(self, **kwargs):
        """
        초기화
        """
        super().__init__(**kwargs)
        
        # 기본 MFA 설정
        self.dictionary_path = None
        self.acoustic_model_path = None
        self.g2p_model_path = None
        
        self.logger.info("Initialized TextGridProcessor")
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        오디오와 텍스트를 정렬하여 TextGrid 파일 생성
        LRS3 데이터는 직접 처리하고, 표준 데이터는 MFA 사용
        """
        start_time = self._log_start("Starting TextGrid processing")
        
        # 컨텍스트에서 필요한 정보 가져오기
        audio_files = context.get('audio_files', [])
        video_files = context.get('video_files', [])
        output_path = context.get('output_path')
        corpus_path = context.get('corpus_path', os.path.join(output_path, 'corpus'))
        config = context.get('config')
        database_type = context.get('database_type')
        if not audio_files:
            self.logger.warning("No audio files found in context")
            return context
        
        # 출력 디렉토리 준비
        textgrid_dir = os.path.join(output_path, 'preprocessed', 'TextGrid')
        self._ensure_directory(textgrid_dir)
        
        # TextGrid 파일 정보 리스트
        textgrid_files = []
        
        # 모든 텍스트 파일 형식 미리 확인
        is_lrs3_corpus = False
        lrs3_files = 0
        standard_files = 0
        
        # for audio_info in audio_files:
        #     audio_id = audio_info.get('id')
        #     transcript_path = audio_info.get('transcript_path')
            
        #     # 텍스트 파일 찾기 시도
        #     if not transcript_path or not os.path.exists(transcript_path):
        #         potential_txt = os.path.join(corpus_path, f"{audio_id}.txt")
        #         if os.path.exists(potential_txt):
        #             transcript_path = potential_txt
            
        #     if transcript_path and os.path.exists(transcript_path):
        #         text_format = self._detect_text_format(transcript_path)
        #         if text_format == 'lrs3':
        #             lrs3_files += 1
        #         else:
        #             standard_files += 1
        
        # # 대부분의 파일이 LRS3 형식이면 전체 말뭉치를 LRS3로 간주
        # if lrs3_files > 0 and lrs3_files >= standard_files:
        #     is_lrs3_corpus = True
        #     self.logger.info(f"Detected LRS3 corpus: {lrs3_files} LRS3 files, {standard_files} standard files")
        if database_type=='lrs3':
            is_lrs3_corpus = True
        
        self.logger.info("Processing corpus as standard format - using MFA")
        
        # MFA에 필요한 파일 확인
        self.dictionary_path = config.get('audio.mfa.dictionary_path', self.dictionary_path)
        self.acoustic_model_path = config.get('audio.mfa.acoustic_model_path', self.acoustic_model_path)
        self.g2p_model_path = config.get('audio.mfa.g2p_model_path', self.g2p_model_path)
        
        if not all([self.dictionary_path, self.acoustic_model_path, self.g2p_model_path]):
            self.logger.error("Missing MFA model paths in configuration")
            return context
        
        if not all([os.path.exists(self.dictionary_path), 
                os.path.exists(self.acoustic_model_path), 
                os.path.exists(self.g2p_model_path)]):
            self.logger.error("One or more MFA model files not found")
            return context
        
        # MFA 실행
        success = self._run_mfa(corpus_path, textgrid_dir,is_lrs3_corpus)
        
        if success:
            # 생성된 TextGrid 파일 확인
            for audio_info in audio_files:
                audio_id = audio_info.get('id')
                tg_file_path = os.path.join(textgrid_dir, f"{audio_id}.TextGrid")
                
                if os.path.exists(tg_file_path):
                    textgrid_files.append({
                        'id': audio_id,
                        'path': tg_file_path,
                        'format': 'standard'
                    })
                else:
                    self.logger.warning(f"TextGrid file not generated for {audio_id}")
    
        # 컨텍스트에 TextGrid 파일 정보 추가
        context['textgrid_files'] = textgrid_files
        
        self._log_end(start_time, f"Generated {len(textgrid_files)} TextGrid files")
        return context
    
    def _process_lrs3_text(self, audio_path: str, text_path: str, output_tg_path: str) -> bool:
        """
        LRS3 형식 텍스트 파일에서 대본을 추출하고 MFA를 실행하여 TextGrid 생성
        
        Args:
            audio_path: 오디오 파일 경로
            text_path: LRS3 텍스트 파일 경로
            output_tg_path: 출력 TextGrid 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 텍스트 파일 읽기
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # # 대본 추출
            # import re
            # transcript_match = re.search(r'Text:\s+(.*?)(?:\s+Conf:|$)', content, re.DOTALL)
            # if not transcript_match:
            #     self.logger.warning(f"Could not extract transcript from LRS3 file: {text_path}")
            #     return False
                
            transcript = content
            
            # 임시 디렉토리 생성
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 오디오 파일 복사
                audio_id = os.path.splitext(os.path.basename(audio_path))[0]
                temp_audio = os.path.join(temp_dir, f"{audio_id}.wav")
                shutil.copy2(audio_path, temp_audio)
                
                # 텍스트 파일 생성 (표준 형식)
                temp_text = os.path.join(temp_dir, f"{audio_id}.txt")
                with open(temp_text, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                
                # 임시 출력 디렉토리
                temp_output = os.path.join(temp_dir, "output")
                os.makedirs(temp_output, exist_ok=True)
                
                # MFA 명령 구성
                mfa_path = "/home/jovyan/store1/miniconda3/envs/tts_env/bin/mfa"
                env = os.environ.copy()
                env["PATH"] = "/home/jovyan/store1/miniconda3/envs/tts_env/bin:" + env["PATH"]
                
                # MFA 실행
                command = [
                    mfa_path, 'align', 
                    '--clean',
                    '--overwrite',
                    '--verbose',
                    '--beam', '100',  # 더 큰 빔 사이즈 사용
                    '--retry_beam', '400',  # 더 큰 재시도 빔 사이즈
                    '--use_mp',  # 멀티프로세싱 활성화
                    '--speaker_characters','11',
                    '--use_threading',  # 멀티스레딩 활성화
                    '--output_format', 'long_textgrid',  # TextGrid 형식 지정
                    temp_dir,
                    self.dictionary_path,
                    self.acoustic_model_path,
                    temp_output
                ]
                
                self.logger.debug(f"Running MFA for file: {audio_id}")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )
                
                stdout, stderr = process.communicate()
                
                # if process.returncode != 0:
                #     self.logger.warning(f"MFA failed for {audio_id}: {stderr}")
                #     # 백업 방법: 단순 TextGrid 생성
                #     return self._create_simple_textgrid(audio_path, transcript, output_tg_path)
                
                # 생성된 TextGrid 파일 확인
                temp_tg = os.path.join(temp_output, f"{audio_id}.TextGrid")
                if os.path.exists(temp_tg):
                    # 출력 디렉토리 생성
                    os.makedirs(os.path.dirname(output_tg_path), exist_ok=True)
                    # TextGrid 파일 복사
                    shutil.copy2(temp_tg, output_tg_path)
                    self.logger.info(f"Created TextGrid using MFA for {audio_id}")
                    return True
                # else:
                #     self.logger.warning(f"MFA did not generate TextGrid for {audio_id}")
                #     # 백업 방법: 단순 TextGrid 생성
                #     return self._create_simple_textgrid(audio_path, transcript, output_tg_path)
                    
        except Exception as e:
            self.logger.error(f"Error processing LRS3 text: {e}")
            self.logger.error(traceback.format_exc())
            return False
    def _find_transcript_paths(self, audio_id: str, corpus_path: str, video_files: List[Dict[str, Any]]) -> List[str]:
        """
        오디오 ID에 해당하는 텍스트 파일 경로들을 찾음
        
        Args:
            audio_id: 오디오 ID
            corpus_path: corpus 디렉토리 경로
            video_files: 비디오 파일 정보 리스트
            
        Returns:
            List[str]: 가능한 텍스트 파일 경로 목록
        """
        paths = [
            # 1. corpus 디렉토리에서 audio_id.txt 찾기
            os.path.join(corpus_path, f"{audio_id}.txt")
        ]
        
        # 2. 비디오 정보 기반으로 찾기
        video_info = next((v for v in video_files if v['id'] == audio_id), None)
        if video_info:
            folder = video_info.get('folder', '')
            index = video_info.get('index', '')
            
            # 가능한 패턴들 추가
            paths.extend([
                os.path.join(corpus_path, f"{folder}_{index}.txt"),  # video_id_index.txt
                os.path.join(corpus_path, f"{index}.txt"),           # index.txt (LRS3 스타일)
                os.path.join(self.root_path, folder, f"{index}.txt") if hasattr(self, 'root_path') else ""  # 원본 위치
            ])
        
        # 빈 경로 제거
        return [p for p in paths if p]

    def _detect_text_format(self, text_path: str) -> str:
        """
        텍스트 파일 형식 감지
        
        Args:
            text_path: 텍스트 파일 경로
            
        Returns:
            str: 텍스트 형식 ('lrs3', 'all_list', 'standard')
        """
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read(200)  # 처음 200자만 읽기
            
            # LRS3 형식 확인
            if 'Text:' in content and ('Conf:' in content or 'WORD START END' in content):
                return 'lrs3'
            
            # All_list.txt 형식 확인 (video_id|text)
            if '|' in content and not content.startswith('Text:'):
                return 'all_list'
                
            # 기본값은 표준 텍스트
            return 'standard'
        except Exception as e:
            self.logger.error(f"Error detecting text format: {e}")
            return 'standard'

    def _run_mfa(self, corpus_dir: str, output_dir: str, is_lrs3_corpus: bool) -> bool:
        """
        Montreal Forced Aligner 실행 - 전체 corpus를 한 번에 처리
        
        Args:
            corpus_dir: 코퍼스 디렉토리 경로 (오디오 및 텍스트 파일 포함)
            output_dir: 출력 디렉토리 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # MFA 명령 구성
            mfa_path = "/home/jovyan/store1/miniconda3/envs/tts_env/bin/mfa"

            env = os.environ.copy()
            env["PATH"] = "/home/jovyan/store1/miniconda3/envs/tts_env/bin:" + env["PATH"]
            
            # 개선된 MFA 명령 - 전체 corpus 처리
            # --speaker_characters 11
            if is_lrs3_corpus:
                command = [
                    mfa_path, 'align', 
                    '--clean',
                    '--use_mp',  # 멀티프로세싱 활성화
                    '--speaker_characters','11',
                    '--use_threading',  # 멀티스레딩 활성화
                    '--output_format', 'long_textgrid',  # TextGrid 형식 지정
                    '--audio_directory', corpus_dir,  # 오디오 디렉토리 지정
                    '--g2p_model_path', self.g2p_model_path,  # G2P 모델 경로
                    corpus_dir,  # corpus 디렉토리
                    self.dictionary_path,  # 사전 경로
                    self.acoustic_model_path,  # 음향 모델 경로
                    output_dir  # 출력 디렉토리
                ]
            else:
                command = [
                    mfa_path, 'align', 
                    '--clean',
                    '--use_mp',  # 멀티프로세싱 활성화
                    '--single_speaker',  # 단일 화자 처리
                    '--use_threading',  # 멀티스레딩 활성화
                    '--output_format', 'long_textgrid',  # TextGrid 형식 지정
                    '--audio_directory', corpus_dir,  # 오디오 디렉토리 지정
                    '--g2p_model_path', self.g2p_model_path,  # G2P 모델 경로
                    corpus_dir,  # corpus 디렉토리
                    self.dictionary_path,  # 사전 경로
                    self.acoustic_model_path,  # 음향 모델 경로
                    output_dir  # 출력 디렉토리
                ]
            
            # MFA 실행
            self.logger.info(f"Running Montreal Forced Aligner for entire corpus: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"MFA failed: {stderr}")
                return False
            
            self.logger.info("MFA alignment completed successfully for entire corpus")
            return True
        except Exception as e:
            self.logger.error(f"Error running MFA: {e}")
            self.logger.error(traceback.format_exc())
            return False