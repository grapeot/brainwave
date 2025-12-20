# x.ai Voice Agent API vs OpenAI Realtime API 对比

## 主要差异

### 1. 端点
- **OpenAI**: `wss://api.openai.com/v1/realtime`
- **x.ai**: `wss://api.x.ai/v1/realtime`

### 2. 认证
- **OpenAI**: 
  - `Authorization: Bearer {api_key}`
  - `OpenAI-Beta: realtime=v1` (必需)
- **x.ai**: 
  - `Authorization: Bearer {api_key}` (仅此)

### 3. 会话配置
- **OpenAI**: 
  - 使用 `session.update` 配置
  - 支持 `modalities` 参数（text, audio）
  - 使用 `input_audio_format: "pcm16"`
  - 使用 `input_audio_transcription.model` 指定转录模型
  - `turn_detection` 可以设置为 `null` 禁用服务器端 VAD
- **x.ai**: 
  - 使用 `session.update` 配置
  - 支持 `voice` 参数（Ara, Rex, Sal, Eve, Leo）
  - 使用 `audio.input.format.type` 和 `audio.input.format.rate` 配置音频格式
  - 支持多种音频格式：`audio/pcm`, `audio/pcmu` (G.711 μ-law), `audio/pcma` (G.711 A-law)
  - `turn_detection` 可以设置为 `{"type": "server_vad"}` 或 `null`

### 4. 音频格式
- **OpenAI**: 
  - PCM16 (Linear16)
  - 默认 24kHz
  - Base64 编码
- **x.ai**: 
  - PCM (Linear16) - 支持多种采样率：8kHz, 16kHz, 21.05kHz, 24kHz, 32kHz, 44.1kHz, 48kHz
  - G.711 μ-law (PCMU) - 8kHz
  - G.711 A-law (PCMA) - 8kHz
  - Base64 编码

### 5. 消息类型
两者消息类型非常相似，但有一些细微差别：

#### 客户端消息
- `session.update` - 两者都有
- `input_audio_buffer.append` - 两者都有
- `input_audio_buffer.commit` - 两者都有（x.ai 仅在 `turn_detection: null` 时可用）
- `input_audio_buffer.clear` - 两者都有
- `conversation.item.create` - 两者都有
- `response.create` - 两者都有

#### 服务器消息
- `session.created` / `conversation.created` - OpenAI 用 `session.created`，x.ai 用 `conversation.created`
- `session.updated` - 两者都有
- `input_audio_buffer.committed` - 两者都有
- `conversation.item.input_audio_transcription.completed` - 两者都有
- `response.created` - 两者都有
- `response.output_audio_transcript.delta` - 两者都有
- `response.output_audio.delta` - 两者都有
- `response.done` - 两者都有

### 6. 功能特性
- **OpenAI**: 
  - 支持工具调用（function calling）
  - 支持多种模型（gpt-4o-realtime-preview, gpt-realtime-mini 等）
  - 支持 ephemeral tokens
- **x.ai**: 
  - 支持工具调用（web_search, x_search, file_search, custom functions）
  - 支持多种语音选项（5种不同的声音）
  - 支持 ephemeral tokens
  - 支持多语言（100+ 语言）

### 7. Turn Detection
- **OpenAI**: 
  - `turn_detection: null` - 禁用服务器端 VAD，使用手动提交
  - 可以配置服务器端 VAD 参数
- **x.ai**: 
  - `turn_detection: {"type": "server_vad"}` - 启用服务器端 VAD
  - `turn_detection: null` - 禁用服务器端 VAD，使用手动提交

## 相似之处

1. **WebSocket 协议**: 两者都使用 WebSocket + JSON 消息
2. **音频编码**: 都使用 Base64 编码的音频数据
3. **消息流程**: 基本流程相似（连接 → 配置 → 发送音频 → 提交 → 获取响应）
4. **实时流式**: 两者都支持实时音频流式传输
5. **转录**: 两者都支持音频转录

## 测试脚本

已创建 `test_xai_voice.py` 脚本用于测试 x.ai Voice Agent API：
- 使用 `test.m4a` 文件
- 使用相同的 prompt (`paraphrase-gpt-realtime-enhanced`)
- 支持手动 turn detection（`turn_detection: null`）
- 输出转录和响应文本

## 运行测试

```bash
# 确保 .env 文件中有 XAI_API_KEY
source py310/bin/activate  # 或使用你的虚拟环境
python test_xai_voice.py
```

