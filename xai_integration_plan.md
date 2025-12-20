# x.ai Voice Agent API 集成计划

## 目标
将 x.ai Voice Agent API 集成到现有的 Brainwave Realtime 系统中，使其可以作为 OpenAI Realtime API 的替代选项。

## 当前架构分析

### 现有组件
1. **`openai_realtime_client.py`** - OpenAI Realtime API 客户端封装
   - 类：`OpenAIRealtimeAudioTextClient`
   - 功能：WebSocket 连接、音频发送、消息处理、handler 注册

2. **`realtime_server.py`** - FastAPI 服务器
   - 端点：
     - `/api/v1/ws` - 主要 WebSocket 端点（实时音频处理）
     - `/api/v1/ws/transcribe` - 转录专用 WebSocket 端点
     - `/api/v1/realtime/session` - 创建 ephemeral session（WebRTC）
   - 使用 `OpenAIRealtimeAudioTextClient` 处理音频

3. **`config.py`** - 配置管理
   - `OPENAI_REALTIME_MODEL`
   - `OPENAI_REALTIME_MODALITIES`
   - `OPENAI_REALTIME_SESSION_TTL_SEC`

4. **前端** (`static/realtime.html`, `static/main.js`)
   - 模型选择下拉框
   - WebSocket 连接和音频流处理

## 集成方案设计

### 方案选择：抽象接口 + 多实现

**推荐方案**：创建一个抽象基类，然后实现两个具体的客户端类。这样可以：
- 保持代码结构清晰
- 最小化对现有代码的修改
- 便于未来添加更多 provider
- 保持向后兼容

### 实施步骤

#### Phase 1: 创建 x.ai 客户端类

**文件**: `xai_realtime_client.py`

**功能**:
- 创建 `XAIRealtimeAudioTextClient` 类
- 参考 `OpenAIRealtimeAudioTextClient` 的接口设计
- 实现相同的公共方法：
  - `__init__(api_key, voice="Ara")`
  - `connect(modalities, session_mode)`
  - `send_audio(audio_data)`
  - `commit_audio()`
  - `clear_audio_buffer()`
  - `register_handler(message_type, handler)`
  - `close()`
- 处理 x.ai 特有的消息类型差异：
  - `conversation.created` vs `session.created`
  - `session.updated` vs `ping`
  - 其他消息类型映射

**关键差异处理**:
- x.ai 不需要 `OpenAI-Beta` header
- x.ai 使用 `voice` 参数而不是 `model`（但可以保留 model 参数用于未来）
- x.ai 的 `turn_detection` 格式：`{"type": "server_vad"}` 或 `null`
- x.ai 的音频格式配置在 `session.update` 中

#### Phase 2: 创建抽象接口（可选但推荐）

**文件**: `realtime_client_base.py` 或直接在现有文件中添加

**选项 A - 抽象基类**:
```python
from abc import ABC, abstractmethod

class RealtimeClientBase(ABC):
    @abstractmethod
    async def connect(self, modalities, session_mode):
        pass
    
    @abstractmethod
    async def send_audio(self, audio_data):
        pass
    
    # ... 其他抽象方法
```

**选项 B - 鸭子类型（更简单）**:
- 不创建抽象基类
- 确保两个客户端类有相同的公共接口
- 通过类型提示和文档说明接口约定

**推荐**: 选项 B，因为 Python 的鸭子类型已经足够，且更简单。

#### Phase 3: 修改配置管理

**文件**: `config.py`

**添加配置**:
```python
# x.ai 配置
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_REALTIME_VOICE = os.getenv("XAI_REALTIME_VOICE", "Ara")  # Ara, Rex, Sal, Eve, Leo
XAI_REALTIME_SAMPLE_RATE = int(os.getenv("XAI_REALTIME_SAMPLE_RATE", "24000"))

# Provider 选择
REALTIME_PROVIDER = os.getenv("REALTIME_PROVIDER", "openai")  # "openai" 或 "xai"
```

#### Phase 4: 修改服务器端点

**文件**: `realtime_server.py`

**修改点**:

1. **导入新客户端**:
```python
from xai_realtime_client import XAIRealtimeAudioTextClient
from config import REALTIME_PROVIDER, XAI_API_KEY, XAI_REALTIME_VOICE
```

2. **创建客户端工厂函数**:
```python
async def create_realtime_client(provider: str = None, model: str = None, voice: str = None):
    """Factory function to create appropriate realtime client"""
    provider = provider or REALTIME_PROVIDER
    
    if provider == "xai":
        api_key = XAI_API_KEY
        if not api_key:
            raise ValueError("XAI_API_KEY not set")
        return XAIRealtimeAudioTextClient(api_key, voice=voice or XAI_REALTIME_VOICE)
    else:  # default to openai
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIRealtimeAudioTextClient(api_key, model=model or OPENAI_REALTIME_MODEL)
```

3. **修改 `/api/v1/ws` 端点**:
   - 在 `initialize_openai` 函数中：
     - 重命名为 `initialize_realtime_client`
     - 使用工厂函数创建客户端
     - 根据 provider 选择不同的客户端
   - 在 WebSocket 消息处理中：
     - 支持 `provider` 参数（可选，默认从配置读取）
     - 支持 `voice` 参数（x.ai 专用）

4. **修改 `/api/v1/ws/transcribe` 端点**:
   - 类似地使用工厂函数
   - 确保转录模式在两个 provider 中都能工作

5. **修改 `/api/v1/realtime/session` 端点**（WebRTC）:
   - 这个端点目前只支持 OpenAI
   - **选项**: 
     - 暂时不修改（x.ai 可能不支持 WebRTC ephemeral tokens）
     - 或者添加 provider 参数，但 x.ai 返回错误时优雅降级

#### Phase 5: 前端修改（可选）

**文件**: `static/realtime.html`, `static/main.js`

**修改点**:

1. **添加 Provider 选择**（可选）:
```html
<label for="providerSelect">Provider:</label>
<select id="providerSelect">
    <option value="openai" selected>OpenAI</option>
    <option value="xai">x.ai (Grok)</option>
</select>
```

2. **修改 WebSocket 消息**:
```javascript
await ws.send(JSON.stringify({ 
    type: 'start_recording', 
    model: selectedModel,
    provider: selectedProvider  // 新增
}));
```

**注意**: 如果不想修改前端，可以在后端根据配置自动选择 provider。

#### Phase 6: 错误处理和兼容性

**需要考虑的问题**:

1. **消息类型差异**:
   - x.ai 使用 `conversation.created`，OpenAI 使用 `session.created`
   - 需要在客户端内部处理这些差异
   - 或者统一消息类型名称（推荐）

2. **音频格式**:
   - 两者都支持 PCM16，24kHz
   - 确保 `AudioProcessor` 的输出格式兼容

3. **错误处理**:
   - API key 缺失时的错误提示
   - 连接失败时的重试逻辑
   - Provider 不支持的功能（如 WebRTC）的优雅降级

4. **日志和调试**:
   - 添加 provider 信息到日志
   - 区分不同 provider 的错误消息

#### Phase 7: 测试

**测试清单**:

1. **单元测试**:
   - `test_xai_realtime_client.py` - 测试 x.ai 客户端
   - 测试消息类型映射
   - 测试连接和断开

2. **集成测试**:
   - 测试 `/api/v1/ws` 端点使用 x.ai
   - 测试 `/api/v1/ws/transcribe` 端点使用 x.ai
   - 测试音频流传输
   - 测试转录功能

3. **端到端测试**:
   - 使用浏览器测试完整流程
   - 测试 provider 切换
   - 测试错误场景

## 文件修改清单

### 新建文件
- [ ] `xai_realtime_client.py` - x.ai 客户端实现
- [ ] `test_xai_realtime_client.py` - x.ai 客户端测试

### 修改文件
- [ ] `config.py` - 添加 x.ai 配置
- [ ] `realtime_server.py` - 集成 x.ai 客户端
- [ ] `static/realtime.html` - 添加 provider 选择（可选）
- [ ] `static/main.js` - 支持 provider 参数（可选）
- [ ] `requirements.txt` - 确保所有依赖已包含

### 文档更新
- [ ] `README.md` - 更新使用说明
- [ ] `.env.example` - 添加 x.ai 配置示例

## 实施优先级

### 高优先级（核心功能）
1. ✅ 创建 `xai_realtime_client.py`
2. ✅ 修改 `config.py` 添加配置
3. ✅ 修改 `realtime_server.py` 的 `/api/v1/ws` 端点
4. ✅ 测试基本功能

### 中优先级（完善功能）
5. 修改 `/api/v1/ws/transcribe` 端点
6. 添加错误处理和日志
7. 添加单元测试

### 低优先级（可选功能）
8. 前端 provider 选择 UI
9. WebRTC 端点支持（如果 x.ai 支持）
10. 性能优化和监控

## 风险评估

### 技术风险
- **低**: x.ai API 与 OpenAI API 非常相似，集成风险低
- **中**: 消息类型差异可能导致 bug，需要充分测试

### 兼容性风险
- **低**: 保持向后兼容，默认使用 OpenAI
- **低**: 现有功能不受影响

### 维护风险
- **中**: 需要维护两套客户端代码
- **低**: 接口统一，维护成本可控

## 后续优化方向

1. **统一接口抽象**: 如果未来添加更多 provider，考虑创建抽象基类
2. **配置管理**: 考虑使用更强大的配置管理库（如 pydantic-settings）
3. **监控和指标**: 添加 provider 使用统计和性能监控
4. **A/B 测试**: 支持同时使用多个 provider 进行对比测试

## 注意事项

1. **API Key 管理**: 确保 `.env` 文件中有 `XAI_API_KEY`
2. **向后兼容**: 默认行为保持不变（使用 OpenAI）
3. **错误处理**: 提供清晰的错误消息，帮助用户诊断问题
4. **文档**: 更新 README，说明如何配置和使用 x.ai provider
5. **测试**: 充分测试两个 provider 的功能，确保行为一致

