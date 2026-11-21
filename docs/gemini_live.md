# Gemini Live 实时观察 MVP

在运行 PvZ 的 Windows 电脑上，持续采集游戏客户区，以最多 1 FPS 发送 JPEG。
默认每轮完成后等待 5 秒再请求判断；推理期间继续发送画面。结果以中文输出到终端和 JSONL。
此入口不暂停、不点击游戏、不读游戏内存，不启动原 Agent 或 WebUI。

## 启动

在仓库根目录执行：

```powershell
python -m pip install -e ".[gemini]"
```

将 `.env.example` 复制为 `.env`（已有文件则编辑，不要覆盖），配置：

```dotenv
GEMINI_API_KEY=你的密钥
GEMINI_LIVE_MODEL=你的账号可用的Live模型ID
WINDOW_TITLE=Plants vs. Zombies
```

模型必须支持 Live 视频输入及 function calling，普通 Gemini 文本模型不能代替。
可按当前官方 SDK 示例尝试 `gemini-3.8-live`，最终以账号实际可用模型为准。

```powershell
python examples/gemini_live_demo.py
# 自定义窗口、判断间隔和输出位置
python examples/gemini_live_demo.py --window "植物大战僵尸" --interval 5 --output action_logs/live.jsonl
```

将游戏保持可见、不要最小化或遮挡；程序不抢焦点。按 Ctrl+C 退出并释放采集资源。
默认日志位于 `action_logs/gemini_live_时间戳.jsonl`，不保存截图或 API 密钥。
画面会传输给 Gemini，并可能产生 API 费用；没有采集麦克风音频。

## 输出和限制

`observation` 记录包含 `phase`、`summary`、`threats`、`suggestions`、`uncertainties`、
`observation_id` 和 `latency_seconds`。延迟从发出判断请求计时，不代表精确的单帧推理延迟：
请求期间持续发送新画面，模型可能结合后续帧。建议只是文字，任何情况下都不执行。

协议使用 AUDIO 输出模态及 `report_game_state` 工具收集结构化结果；音频不播放。
若模型只说话而未调用工具，会记录 warning，不将语音或文本假装成结构化结果。
使用服务端上下文压缩与恢复句柄；连接异常最多重试 3 次，判断超时后重建新会话。
这是有界重试的 MVP，不是无限值守服务。识别准确率和费用需实际游戏验证。

## 验证

```powershell
python -m unittest discover -s tests -p "test_live_observer.py"
```

模拟测试覆盖结果校验、重复/过期报告和多轮接收；不需要 API Key 或 Windows。
实机验收：观察至少两轮结果、切换选卡/战斗场景检查判断，并按 Ctrl+C 检查退出。

协议参考：[SDK 入门](https://aistudio.google.com/docs/live-api/get-started-sdk)、
[会话管理](https://aistudio.google.com/docs/live-api/session-management)。
