# Check how the GTE model loads on CPU.

import importlib.util
import json
from pathlib import Path
import traceback
from transformers.dynamic_module_utils import check_imports
from fetch_assets import ROOT, sha256


def main():
    source = ROOT / ".assets/model/modeling_qwen.py"
    expected = next(
        item["sha256"]
        for item in json.loads((ROOT / "assets.json").read_text())["assets"]
        if item["path"] == "model/modeling_qwen.py"
    )
    assert sha256(source) == expected
    output = ROOT / "verification/loader"
    output.mkdir(parents=True, exist_ok=True)
    try:
        check_imports(str(source))
    except ImportError as error:
        assert "flash_attn" in str(error)
        failure = {
            "type": type(error).__name__,
            "message": str(error),
        }
        (output / "static_import_check.log").write_text(
            traceback.format_exc().replace(str(ROOT), "<PACKAGE_ROOT>"),
            encoding="utf-8",
        )
    else:
        raise AssertionError(
            "Expected the locked Transformers static scanner to request the optional GPU dependency"
        )
    specification = importlib.util.spec_from_file_location(
        "verified_public_gte_loading_probe", source
    )
    implementation = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(implementation)
    assert hasattr(implementation, "Qwen2Model")
    result = {
        "first_wrapper_process_exit_code": 1,
        "static_check": failure,
        "public_source_sha256": expected,
        "scope": "Changed this package loading wrapper to direct Python import of the checked public class; its own optional FlashAttention condition and eager CPU code remain intact.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
