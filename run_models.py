# Run a model's small example from the repository root.

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def load_models():
    records = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    for record in records:
        folder = (ROOT / record["path"]).resolve()
        if ROOT not in folder.parents or not (folder / "pixi.toml").is_file():
            raise ValueError("模型目录无效：" + record["path"])
    return records


def main():
    parser = argparse.ArgumentParser(
        description="下载公开示例所需文件，并调用各包的小数据运行入口。"
    )
    parser.add_argument(
        "--model",
        default="gram",
        help="模型名称；all 运行每篇论文的默认入口，all-routes 包括其他输入示例。",
    )
    parser.add_argument(
        "--list", action="store_true", help="查看模型名称、输入和运行范围。"
    )
    args = parser.parse_args()
    models = load_models()
    if args.list:
        for model in models:
            print(
                f"{model['id']:22} [{model['paper']}] {model['name']} | {model['input']} | {model['scope']}"
            )
        return 0
    if args.model == "all":
        selected = [model for model in models if model.get("default", True)]
    elif args.model == "all-routes":
        selected = models
    else:
        selected = [model for model in models if model["id"] == args.model]
    if not selected:
        parser.error("没有找到这个模型。运行 --list 查看可用名称。")
    pixi = shutil.which("pixi")
    if pixi is None:
        parser.error("请先安装 Pixi：https://pixi.sh/latest/installation/")
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / ".run" / "smoke" / run_id
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, model in enumerate(selected, 1):
        command = [
            pixi,
            "run",
            "--locked",
            "--manifest-path",
            str(ROOT / model["path"] / "pixi.toml"),
            model.get("task", "smoke"),
        ]
        print(
            f"\n[{index}/{len(selected)}] {model['name']}：{model['input']}", flush=True
        )
        print("运行范围：" + model["scope"], flush=True)
        started = time.monotonic()
        log_path = output / (model["id"] + ".log")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        cert_dir = env.get("SSL_CERT_DIR")
        if cert_dir:
            cert_path = Path(cert_dir)
            if cert_path.is_relative_to(ROOT / ".pixi") and not any(
                cert_path.glob("*")
            ):
                env.pop("SSL_CERT_DIR")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT / model["path"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        results.append(
            {
                "model": model["id"],
                "path": model["path"],
                "returncode": code,
                "seconds": round(time.monotonic() - started, 2),
                "log": str(log_path.relative_to(ROOT)),
                "input": model["input"],
                "scope": model["scope"],
            }
        )
        (output / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{model['name']}：{'完成' if code == 0 else '运行中断'}。记录：{log_path}",
            flush=True,
        )
    print(f"\n运行记录保存在 {output}")
    return int(any(result["returncode"] for result in results))


if __name__ == "__main__":
    sys.exit(main())
