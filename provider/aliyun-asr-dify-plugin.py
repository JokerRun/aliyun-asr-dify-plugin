from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
import dashscope
from dashscope.audio.asr import Transcription
from dashscope.models import Models
from http import HTTPStatus


class AliyunAsrDifyPluginProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            # 检查API Key是否提供
            if "api_key" not in credentials or not credentials.get("api_key"):
                raise ToolProviderCredentialValidationError("API Key is required.")
            
            # 设置API Key
            api_key = credentials.get("api_key")
            dashscope.api_key = api_key
            
            # 测试API Key是否有效，调用一个简单的查询
            # 由于我们不能实际上传音频文件进行测试，只能校验API能否被调用
            try:
                # 使用Models类的list方法验证API密钥
                response = Models.list(api_key=api_key)
                if response.status_code != HTTPStatus.OK:
                    raise ToolProviderCredentialValidationError(f"Failed to validate API Key: {response.message}")
            except Exception as e:
                raise ToolProviderCredentialValidationError(f"Failed to connect to Aliyun Dashscope: {str(e)}")
                
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
