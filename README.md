# 阿里云语音识别 Dify 插件

**作者:** jerry
**版本:** 0.0.2
**插件类型:** 语音识别工具

## 描述
阿里云语音识别插件（Aliyun ASR Plugin）是一个基于阿里云 Paraformer 语音识别服务的 Dify 插件，可以将会议录音文件转录为文本脚本。该插件支持多语种识别（中文、英文、日语等）和说话人分离（Speaker Diarization）功能，适用于会议记录、访谈转写等场景。

## 功能特点
- 支持多种音频格式：mp3、wav、m4a、aac、flac等
- 支持多语种识别：中文（含多种方言）、英文、日语、韩语等
- 支持说话人分离：可识别不同说话人并在转录结果中添加标签
- 支持时间戳：为每个句子提供精确的开始和结束时间
- 高准确率：采用阿里云先进的 Paraformer 语音识别模型

## 配置说明
1. **API Key**：阿里云 Dashscope API 密钥，可在[阿里云 Dashscope 控制台](https://dashscope.console.aliyun.com/apiKey)申请获取

## 参数说明
- **音频文件URL**（必选）：要转录的音频文件的URL，必须是可公开访问的HTTP/HTTPS链接
- **识别模型**（可选）：指定使用的ASR模型，默认为"paraformer-v2"
- **启用说话人分离**（可选）：是否识别不同说话人，默认为false
- **说话人数量**（可选）：预期的说话人数量，仅在启用说话人分离时使用，默认为2

## 支持的模型
- **paraformer-v2**：多语种语音识别模型（推荐使用），支持中文、英文、日语等
- **paraformer-8k-v2**：适用于8kHz电话场景的中文语音识别模型
- **paraformer-v1**：中英文语音识别模型
- **paraformer-8k-v1**：适用于8kHz电话场景的中文语音识别模型
- **paraformer-mtl-v1**：多语言语音识别模型

## 使用示例
```
请将此音频文件转录为文本：https://example.com/meeting.mp3
```

高级使用（带说话人分离）：
```
请将此会议录音转录为文本，并识别不同的发言者：https://example.com/meeting.mp3，预计有3位发言者
```

## 注意事项
- 音频文件必须可通过公网访问（支持HTTP/HTTPS协议）
- 文件大小不超过2GB，时长在12小时以内
- 识别结果包含时间戳、原始文本和格式化后的脚本



