# Check the fix for empty level tables.

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

for name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "2"
os.environ.setdefault(
    "THEANO_FLAGS", "blas.ldflags=,cxx=,mode=FAST_COMPILE,linker=py,floatX=float64"
)
import numpy as np
import theano
import theano.tensor as T
from prepare_author import ROOT, prepare


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tree-prefix", default=str(ROOT / ".private/current/inputs/tree")
    )
    args = parser.parse_args()
    output = ROOT / "verification/compatibility"
    output.mkdir(parents=True, exist_ok=True)
    original_path = ROOT / ".run/original_python3_gram.py"
    runtime = prepare(repair_empty=False)
    shutil.copyfile(runtime / "gram.py", original_path)
    original = module(original_path, "gram_original_compatible")
    runtime = prepare(repair_empty=True)
    fixed = module(runtime / "gram.py", "gram_fixed_empty")
    tree_checks = []
    empty_arrays = None
    for level in range(1, 6):
        path = args.tree_prefix + ".level%d.pk" % level
        leaves, ancestors = fixed.build_tree(path)
        check = {
            "level": level,
            "shape": list(leaves.shape),
            "dtype": str(leaves.dtype),
        }
        if len(leaves):
            old_leaves, old_ancestors = original.build_tree(path)
            check["nonempty_values_and_dtypes_identical"] = (
                np.array_equal(old_leaves, leaves)
                and np.array_equal(old_ancestors, ancestors)
                and old_leaves.dtype == leaves.dtype
                and old_ancestors.dtype == ancestors.dtype
            )
            assert check["nonempty_values_and_dtypes_identical"]
        else:
            try:
                original.build_tree(path)
            except IndexError as error:
                check["original_exception"] = type(error).__name__ + ": " + str(error)
            else:
                raise AssertionError("Expected original empty-table failure")
            assert leaves.shape == ancestors.shape == (0, level + 1)
            empty_arrays = leaves, ancestors
        tree_checks.append(check)
    assert empty_arrays is not None
    rng = np.random.RandomState(1)
    parameters = {
        "W_emb": theano.shared(rng.normal(size=(5, 128))),
        "W_attention": theano.shared(rng.normal(size=(256, 128))),
        "b_attention": theano.shared(rng.normal(size=128)),
        "v_attention": theano.shared(rng.normal(size=128)),
    }
    attention = fixed.generate_attention(
        parameters, theano.shared(empty_arrays[0]), theano.shared(empty_arrays[1])
    )
    embedding = (
        parameters["W_emb"][theano.shared(empty_arrays[1])] * attention[:, :, None]
    ).sum(axis=1)
    nonempty = T.constant(rng.normal(size=(3, 128)))
    concatenation = T.concatenate([embedding, nonempty], axis=0)
    gradient = T.grad(embedding.sum(), parameters["W_emb"])
    function = theano.function(
        [], [attention, embedding, concatenation, gradient], mode="FAST_COMPILE"
    )
    attn, emb, concatenated, grad = function()
    assert attn.shape == empty_arrays[0].shape and emb.shape == (0, 128)
    assert np.array_equal(concatenated, nonempty.data) and np.count_nonzero(grad) == 0
    assert np.isfinite(grad).all()
    before = ast.parse(original_path.read_text(encoding="utf-8"))
    after = ast.parse((runtime / "gram.py").read_text(encoding="utf-8"))
    old_defs = {
        node.name: ast.dump(node, include_attributes=False)
        for node in before.body
        if isinstance(node, ast.FunctionDef)
    }
    new_defs = {
        node.name: ast.dump(node, include_attributes=False)
        for node in after.body
        if isinstance(node, ast.FunctionDef)
    }
    unchanged = {
        name: old_defs[name] == new_defs[name]
        for name in old_defs
        if name != "build_tree"
    }
    assert all(unchanged.values())
    result = {
        "tree_tables": tree_checks,
        "empty_attention_shape": list(attn.shape),
        "empty_embedding_shape": list(emb.shape),
        "other_author_functions_identical": unchanged,
        "added_patients_or_nodes": 0,
        "python3_conversion": "Standard lib2to3 syntax/iterator conversion; no division or random-stream replacement.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
