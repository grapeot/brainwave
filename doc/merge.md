# Merge Plan: Local ↔ Remote Synchronization

## 当前状态

### Git 状态
- **Local branch**: `master`
- **Remote branch**: `origin/master`
- **Divergence**: Local 有 3 个 commit，Remote 有 4 个 commit
- **Base commit**: `2c7244b` (Merge pull request #5)

### 本地未提交的更改
- Modified: `openai_realtime_client.py`, `prompts.py`, `realtime_server.py`, `requirements.txt`, `static/main.js`, `static/realtime.html`, `static/style.css`, `tests/test_openai_realtime_client.py`, `README.md`, `.gitignore`
- Untracked: `config.py`, `static/webrtc.html`, `static/webrtc.js`
- 测试文件（不应提交）: `instructions.m4a`, `instructions.wav`
- 已删除: `dev_local_storage.md`, `static/index_test.html`, `static/index_test.js` (内容已合并到 README.md)

---

## Remote → Local: 需要合并的 Remote 功能

### PR #10: Fix Bluetooth Microphone Bug
- **Commit**: `da04b71`
- **影响文件**: 可能涉及音频处理相关代码
- **合并策略**: 直接合并，检查是否有冲突

### PR #7: Use Enhanced Realtime Prompt
- **Commits**: 
  - `86302b9`: Use enhanced realtime paraphrase prompt
  - `8f489b0`: Default realtime client to gpt-realtime
  - `7eb4e6a`: Enforce realtime transcript prefix and gate streaming
  - `61c264b`: Update realtime prompt examples for prefix
- **影响文件**: 
  - `prompts.py` - 需要合并 prompt 更改
  - `openai_realtime_client.py` - 默认模型更改
  - `realtime_server.py` - 前缀处理逻辑
- **合并策略**: 
  - `prompts.py`: 需要手动合并，本地已有 `gemini-transcription` prompt
  - `openai_realtime_client.py`: 检查默认模型设置，本地已支持模型参数
  - `realtime_server.py`: 需要合并前缀处理逻辑，可能与本地 WebRTC 实现有冲突

### PR #5: Fix Audio Upload Timing Issue
- **Commit**: `5a3da06`
- **影响文件**: 音频处理相关
- **合并策略**: 直接合并

### PR #4: Websockets Version Requirement
- **Commit**: `925d239`
- **影响文件**: `requirements.txt`
- **合并策略**: 合并到 `requirements.txt`

---

## Local → Remote: 需要推送到 Remote 的功能

### 1. Gemini Integration
- **文件**: 
  - `gemini_client.py` (新增)
  - `static/transcribe.html` (新增)
  - `static/transcribe.js` (新增)
- **后端端点**: `/api/v1/transcribe_gemini` (SSE)
- **状态**: 完整实现，需要提交
- **合并策略**: 直接添加，无冲突

### 2. WebRTC Implementation
- **文件**: 
  - `static/webrtc.html` (新增)
  - `static/webrtc.js` (新增)
  - `realtime_server.py` (新增 `/api/v1/realtime/session` 端点)
- **功能**: 
  - WebRTC 直接连接 OpenAI Realtime API
  - Ephemeral session token 管理
  - 模型选择支持
- **状态**: 完整实现，需要提交
- **合并策略**: 
  - 新文件直接添加
  - `realtime_server.py` 需要与 Remote 的更改合并（前缀处理逻辑）

### 3. Local Storage / IndexedDB Replay Feature
- **文件**: 
  - `README.md` (已更新，包含 Local Storage 功能文档)
- **功能**: 
  - 浏览器端录音重放功能文档
  - IndexedDB 存储音频 chunks 的设计说明
- **状态**: 文档已合并到 README.md
- **合并策略**: README.md 的更改需要合并
- **注意**: `index_test.html` 和 `index_test.js` 测试文件已删除，功能文档已整合到 README

### 4. Model Selection Feature
- **文件**: 
  - `static/realtime.html` (修改)
  - `static/webrtc.html` (修改)
  - `static/main.js` (修改)
  - `static/webrtc.js` (修改)
  - `realtime_server.py` (修改 - WebSocket 和 WebRTC 端点都支持模型参数)
- **功能**: 下拉列表选择模型（gpt-realtime / gpt-realtime-mini-2025-12-15）
- **状态**: 完整实现，需要提交
- **合并策略**: 
  - HTML 文件：直接添加下拉列表
  - JavaScript 文件：需要与 Remote 的更改合并
  - `realtime_server.py`: 需要与 Remote 的前缀处理逻辑合并

### 5. Configuration Management
- **文件**: `config.py` (新增)
- **功能**: 集中管理配置（模型、模态等）
- **状态**: 需要提交
- **合并策略**: 直接添加

---

## 合并计划（执行顺序）

### Phase 1: 准备阶段
1. ✅ Fetch remote: `git fetch origin`
2. ✅ 分析差异: 已完成
3. ⏳ 更新 `.gitignore` 排除测试文件
4. ⏳ 提交本地未提交的更改（不包括测试文件）

### Phase 2: 合并 Remote → Local
1. **创建合并分支**: `git checkout -b merge-remote-to-local`
2. **合并 Remote**: `git merge origin/master`
3. **解决冲突**:
   - `prompts.py`: 合并 `paraphrase-gpt-realtime-enhanced` prompt，保留 `gemini-transcription`
   - `openai_realtime_client.py`: 确保默认模型和模型参数支持都保留
   - `realtime_server.py`: 
     - 合并前缀处理逻辑（PR #7）
     - 合并音频上传时序修复（PR #5）
     - 保留 WebRTC 端点和模型选择功能
   - `requirements.txt`: 合并 websockets 版本要求
4. **测试**: 确保所有功能正常工作

### Phase 3: 准备推送到 Remote
1. **创建推送分支**: `git checkout -b merge-local-to-remote`
2. **确保所有本地功能已提交**:
   - Gemini integration
   - WebRTC implementation
   - Local storage replay
   - Model selection
   - Configuration management
3. **更新文档**: 如有必要

### Phase 4: 最终合并
1. **合并到 master**: `git checkout master && git merge merge-remote-to-local`
2. **测试**: 全面测试所有功能
3. **推送到 Remote**: `git push origin master`

---

## 潜在冲突点

### 1. `realtime_server.py`
- **Remote**: 前缀处理逻辑、音频上传时序修复
- **Local**: WebRTC 端点、模型选择、Gemini 端点
- **解决**: 需要仔细合并，确保所有功能都保留

### 2. `prompts.py`
- **Remote**: `paraphrase-gpt-realtime-enhanced` prompt 更新
- **Local**: `gemini-transcription` prompt 新增
- **解决**: 两个 prompt 都需要保留

### 3. `openai_realtime_client.py`
- **Remote**: 默认模型改为 `gpt-realtime`
- **Local**: 支持模型参数、transcription mode
- **解决**: 确保默认模型和参数化都保留

### 4. `static/main.js`
- **Remote**: 可能有音频处理相关的 bug fix
- **Local**: 模型选择功能
- **解决**: 需要仔细合并

---

## 需要排除的文件

以下文件不应提交到 git：

```
# 测试音频文件
instructions.m4a
instructions.wav
test.m4a

# 临时文件（如果存在）
*.tmp
*.temp
scratchpad.*
```

建议更新 `.gitignore`:
```gitignore
*.m4a
*.wav
*.tmp
*.temp
scratchpad.*
```

---

## 测试清单

合并后需要测试：

- [ ] WebSocket 版本录音功能（`/api/v1/ws`）
- [ ] WebRTC 版本录音功能（`/api/v1/realtime/session`）
- [ ] Gemini 转录功能（`/api/v1/transcribe_gemini`）
- [ ] 模型选择功能（所有页面）
- [ ] Local storage 重放功能文档（已在 README.md 中）
- [ ] 前缀处理逻辑（PR #7）
- [ ] 音频上传时序修复（PR #5）
- [ ] Bluetooth microphone bug fix（PR #10）

---

## 注意事项

1. **不要直接 merge**：按照计划逐步执行
2. **保留所有功能**：确保 Remote 和 Local 的功能都保留
3. **测试优先**：每个阶段都要充分测试
4. **文档更新**：如有必要，更新 README 或其他文档

---

## 执行命令参考

```bash
# Phase 1: 准备
git fetch origin
# 更新 .gitignore
git add .gitignore
git commit -m "Update .gitignore to exclude test audio files"

# Phase 2: 合并 Remote → Local
git checkout -b merge-remote-to-local
git merge origin/master
# 解决冲突...
git add .
git commit -m "Merge remote changes: PR #4, #5, #7, #10"

# Phase 3: 准备推送到 Remote
git checkout master
git checkout -b merge-local-to-remote
# 确保所有本地功能已提交
git add config.py static/webrtc.* static/index_test.* static/transcribe.* gemini_client.py
git commit -m "Add WebRTC, local storage replay, Gemini integration, and model selection"

# Phase 4: 最终合并
git checkout master
git merge merge-remote-to-local
# 测试...
git push origin master
```

