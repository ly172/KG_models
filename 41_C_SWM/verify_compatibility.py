# Check API errors and numerical compatibility.

import ast
import importlib.util
import json
import os
from pathlib import Path

from prepare import ROOT, prepare, sha256

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "2"


def load_utils(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    source = prepare()
    import numpy as np
    import torch
    from skimage.draw import disk

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    original = load_utils(ROOT / "upstream/utils.py", "original_cswm_utils")
    patched = load_utils(source / "utils.py", "patched_cswm_utils")
    actions = torch.arange(20, dtype=torch.int32)
    try:
        original.to_one_hot(actions, 20)
    except RuntimeError as error:
        dtype_error = str(error)
    else:
        raise AssertionError("Expected the original int32 index failure")
    assert "Expected dtype int64 for index" in dtype_error
    original_encoded = original.to_one_hot(actions.long(), 20)
    patched_encoded = patched.to_one_hot(actions, 20)
    assert actions.tolist() == actions.long().tolist() == list(range(20))
    assert torch.equal(original_encoded, patched_encoded)
    assert torch.equal(patched_encoded, torch.eye(20))
    dataset = original.StateTransitionsDataset(
        str(ROOT / "outputs/quick/data/shapes_train.h5")
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0
    )
    iterator = iter(loader)
    try:
        iterator.next()
    except AttributeError as error:
        iterator_error = str(error)
    else:
        raise AssertionError("Expected the original iterator API failure")
    assert "has no attribute 'next'" in iterator_error
    batch = next(iterator)
    assert tuple(batch[0].shape[1:]) == (3, 50, 50)
    reference = ROOT / "verification/reference/skimage_0_15_draw.txt"
    reference_info = json.loads(
        reference.with_name("source.json").read_text(encoding="utf-8")
    )
    assert sha256(reference) == reference_info["sha256"]
    tree = ast.parse(reference.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("circle", "ellipse", "_ellipse_in_shape")
    ]
    namespace = {"np": np}
    exec(compile(tree, str(reference), "exec"), namespace)
    for row in range(5):
        for col in range(5):
            center = (row * 10 + 5, col * 10 + 5)
            rr, cc = disk(center, 5, shape=(50, 50))
            old_rr, old_cc = namespace["circle"](*center, 5, (50, 50, 3))
            assert np.array_equal(rr, old_rr) and np.array_equal(cc, old_cc)
    manifest = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    preserved = {
        name: sha256(source / name) == manifest["files"][name]
        for name in ["modules.py", "data_gen/env.py", "envs/__init__.py"]
    }
    assert all(preserved.values())
    run = json.loads((ROOT / "outputs/quick/result.json").read_text(encoding="utf-8"))
    assert run["status"] == "passed" and all(
        step["exit_code"] == 0 for step in run["steps"]
    )
    output = ROOT / "verification/compatibility"
    output.mkdir(parents=True, exist_ok=True)
    prior = []
    for name, marker, code, end in [
        ("01_original_environment.log", "circle", 1, "natural_exit"),
        ("02_circle_name_fixed.log", "NoneType", 1, "natural_exit"),
        ("03_environment_fixed.log", "iter 0", 0, "natural_exit"),
        (
            "04_original_train.log",
            "bootstrapping phase",
            None,
            "terminated_after_exception_during_worker_cleanup",
        ),
    ]:
        path = output / name
        if path.exists():
            assert marker in path.read_text(encoding="utf-8")
            prior.append(
                {
                    "log": name,
                    "marker_present": marker,
                    "exit_code": code,
                    "process_end": end,
                }
            )
    result = {
        "status": "passed",
        "scope": "Numerical and API compatibility checks; original full-model training/evaluation was run separately by reproduce.py.",
        "prior_actual_entrypoint_probes": prior,
        "original_int32_error": dtype_error,
        "original_iterator_error_on_native_dataset": iterator_error,
        "corrected_iterator_batch_shape": list(batch[0].shape),
        "native_training_transitions": len(dataset),
        "model_and_data_generator_files_exactly_original": preserved,
        "old_dependency_source_sha256": sha256(reference),
        "full_model_stages_naturally_exited_zero": [
            step["stage"] for step in run["steps"]
        ],
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
