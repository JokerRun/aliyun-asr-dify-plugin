from collections.abc import Generator
from typing import Any, Dict, Optional
import time
import json
import requests

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
import dashscope
from dashscope.audio.asr import Transcription
from http import HTTPStatus

class TranscribeAudioTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """
        将音频文件转录为文本
        """
        # 获取参数
        file_url = tool_parameters.get("file_url", "")
        model = tool_parameters.get("model", "paraformer-v2")
        diarization_enabled = tool_parameters.get("diarization_enabled", False)
        speaker_count = tool_parameters.get("speaker_count", 2)
        
        # 验证参数
        if not file_url:
            yield self.create_text_message("音频文件URL是必需的。")
            return
        
        try:
            # 获取API密钥
            api_key = self.runtime.credentials.get("api_key")
            if not api_key:
                yield self.create_text_message("需要Aliyun Dashscope API密钥。")
                return
            print("=======api_key readed=======")
            # 设置API密钥
            dashscope.api_key = api_key
            
            # 构建请求参数
            params = {
                "model": model,
                "file_urls": [file_url]
            }
            
            # 如果启用了说话人分离，添加相关参数
            if diarization_enabled:
                params["diarization_enabled"] = True
                params["speaker_count"] = speaker_count
            
            # 提交转录任务
            yield self.create_text_message(f"正在提交音频转录任务，使用模型: {model}...\n\n")
            response = Transcription.async_call(**params)
            print(response)
            if response.status_code != HTTPStatus.OK:
                error_msg = f"提交转录任务失败: {response.message}"
                yield self.create_text_message(error_msg)
                return
            
            # 获取任务ID
            task_id = response.output["task_id"]
            # yield self.create_text_message(f"转录任务已提交，任务ID: {task_id}，正在等待结果...\n\n")
            
            # 等待任务完成
            final_response = Transcription.wait(task_id)
            yield self.create_text_message(f">>>音频转录已完成:final_response >>> \n {final_response} \n <<<音频转录已完成:final_response <<< ")
            
            if final_response.status_code != HTTPStatus.OK:
                error_msg = f"转录任务失败: {final_response.message}"
                yield self.create_text_message(error_msg)
                return
            
            # 处理结果
            result = self._format_result(final_response.output, diarization_enabled)
            
            # 返回结果
            yield self.create_text_message(f">>>转录脚本解析已完成 >>> \n {result} \n <<<转录脚本解析已完成 <<< ")
            # 返回格式化的纯文本结果
            yield self.create_text_message(result["formatted_transcript"])
            yield self.create_json_message(result)
            
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"转录过程中发生错误: {str(e)}\n{error_trace}")
            yield self.create_text_message(f"转录过程中发生错误: {str(e)}\n{error_trace}")
            return
    
    def _download_transcription_result(self, url: str) -> Dict:
        """
        下载并解析转录URL中的结果
        """
        try:
            print(f"正在下载转录结果: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"下载转录结果失败: {response.status_code}")
                return {}
        except Exception as e:
            print(f"下载转录结果时出错: {str(e)}")
            return {}
    
    def _format_result(self, result_output: Dict, diarization_enabled: bool) -> Dict:
        """
        格式化转录结果
        """
        try:
            print(f"开始处理转录结果: {result_output}")
            formatted_transcript = ""
            raw_transcript = ""
            sentences = []
            
            # 获取结果数据
            transcription_url = None
            
            # 检查是否有结果文件的URL
            if 'results' in result_output and len(result_output['results']) > 0:
                first_result = result_output['results'][0]
                if 'transcription_url' in first_result:
                    transcription_url = first_result['transcription_url']
                    
            print(f"转录结果URL: {transcription_url}")
            
            # 如果有转录URL，下载并解析结果
            if transcription_url:
                transcription_result = self._download_transcription_result(transcription_url)
                print(f"下载的转录结果: {transcription_result}")
                
                # 检查是否有transcripts字段
                if 'transcripts' in transcription_result and len(transcription_result['transcripts']) > 0:
                    # 获取第一个转录结果
                    transcript = transcription_result['transcripts'][0]
                    
                    # 获取完整文本
                    raw_transcript = transcript.get('text', '')
                    
                    # 获取句子列表
                    sentences_list = transcript.get('sentences', [])
                    
                    # 获取音频时长
                    duration = transcript.get('content_duration_in_milliseconds', 0)
                    
                    # 处理句子
                    if diarization_enabled and 'speaker' in sentences_list[0]:
                        # 处理带说话人分离的结果
                        current_speaker = None
                        
                        for sentence in sentences_list:
                            speaker = sentence.get('speaker', '')
                            text = sentence.get('text', '').strip()
                            start_time = sentence.get('begin_time', 0)
                            end_time = sentence.get('end_time', 0)
                            
                            # 格式化时间
                            start_formatted = self._format_time(start_time)
                            end_formatted = self._format_time(end_time)
                            
                            # 如果说话人变化，添加新的说话人标签
                            if speaker != current_speaker:
                                current_speaker = speaker
                                formatted_transcript += f"\n说话人 {speaker} [{start_formatted}]：{text}\n"
                            else:
                                formatted_transcript += f"{text} "
                            
                            sentences.append({
                                "speaker": speaker,
                                "text": text,
                                "begin_time": start_time,
                                "end_time": end_time,
                                "begin_time_formatted": start_formatted,
                                "end_time_formatted": end_formatted
                            })
                    else:
                        # 处理普通转录结果
                        for sentence in sentences_list:
                            text = sentence.get('text', '').strip()
                            start_time = sentence.get('begin_time', 0)
                            end_time = sentence.get('end_time', 0)
                            
                            # 格式化时间
                            start_formatted = self._format_time(start_time)
                            end_formatted = self._format_time(end_time)
                            
                            formatted_transcript += f"[{start_formatted} - {end_formatted}] {text}\n"
                            
                            sentences.append({
                                "text": text,
                                "begin_time": start_time,
                                "end_time": end_time,
                                "begin_time_formatted": start_formatted,
                                "end_time_formatted": end_formatted
                            })
                    
                    # 返回格式化的结果
                    return {
                        "formatted_transcript": formatted_transcript.strip(),
                        "raw_transcript": raw_transcript.strip(),
                        "sentences": sentences,
                        "transcription_url": transcription_url,
                        "duration": duration,
                        "duration_formatted": self._format_time(duration)
                    }
            
            # 如果没有找到转录URL或解析失败，返回空结果
            return {
                "formatted_transcript": "未能获取转录结果",
                "raw_transcript": "",
                "sentences": [],
                "transcription_url": transcription_url,
                "duration": 0,
                "duration_formatted": "00:00:00"
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"格式化转录结果时出错: {str(e)}\n{error_trace}")
            return {
                "error": str(e),
                "error_trace": error_trace,
                "formatted_transcript": "处理转录结果时出错",
                "raw_transcript": "",
                "sentences": [],
                "transcription_url": None
            }
    
    def _format_time(self, milliseconds: float) -> str:
        """
        将毫秒转换为可读格式的时间
        """
        if not milliseconds:
            return "00:00:00"
        
        total_seconds = int(milliseconds / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours:02}:{minutes:02}:{seconds:02}" 