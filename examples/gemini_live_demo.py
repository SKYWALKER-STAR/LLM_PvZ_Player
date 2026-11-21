"""Observe PvZ or try verified planting with Gemini Live on Windows."""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="Gemini Live 游戏观察与种植")
    parser.add_argument("--window", help="游戏窗口标题关键词；默认使用 WINDOW_TITLE")
    parser.add_argument("--model", help="覆盖 GEMINI_LIVE_MODEL")
    parser.add_argument("--interval", type=float, default=5, help="本轮结束后等待秒数，默认 5")
    parser.add_argument("--timeout", type=float, default=30, help="单轮判断超时秒数，默认 30")
    parser.add_argument("--scale", type=float, default=1, help="截图缩放比例 (0, 1]")
    parser.add_argument("--output", type=Path, help="JSONL 日志路径")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="提出并校验种植动作，只记录结果")
    mode.add_argument("--play", action="store_true", help="执行通过校验的种植动作")
    args = parser.parse_args()
    if args.interval < 1 or args.timeout < 5 or not 0 < args.scale <= 1:
        parser.error("interval 必须 >=1，timeout 必须 >=5，scale 必须在 (0, 1] 内")
    if sys.platform != "win32":
        parser.error("当前窗口采集仅支持 Windows，请在运行 PvZ 的 Windows 电脑启动")

    from vlm_game_agent.config.settings import Settings
    from vlm_game_agent.live.observer import LiveObserver, WindowSource

    settings = Settings()
    title = args.window or settings.window_title
    model = args.model or settings.gemini_live_model
    if not title or not settings.gemini_api_key or not model:
        parser.error("请设置 WINDOW_TITLE、GEMINI_API_KEY、GEMINI_LIVE_MODEL（或使用对应参数）")
    try:
        import google.genai  # noqa: F401
    except ImportError:
        parser.error('缺少 Gemini SDK，请执行 python -m pip install -e ".[gemini]"')
    output = args.output or ROOT / "action_logs" / f"gemini_live_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    source = WindowSource(title, args.scale)
    planter = None
    if args.dry_run or args.play:
        from vlm_game_agent.live.planting import LivePlanter

        planter = LivePlanter(source, execute=args.play)
    observer = LiveObserver(source, output, args.interval, args.timeout, planter=planter)
    print(f"保持游戏窗口可见、不要最小化或遮挡。Ctrl+C 停止。日志：{output}")
    try:
        asyncio.run(observer.run(settings.gemini_api_key, model))
    except KeyboardInterrupt:
        print("观察已停止。")
    except Exception as exc:
        print(f"观察失败 ({type(exc).__name__})：请检查窗口、网络、模型权限和配置。", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if planter is not None:
            planter.close()


if __name__ == "__main__":
    main()
