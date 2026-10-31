# Run KG-TREAT on artificial inputs.

from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"
import copy, hashlib, json, platform, subprocess, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from prepare import prepare, ROOT


def main():
    start = time.perf_counter()
    compatible = prepare()
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20261029)
    np.random.seed(20261029)
    original_env = dict(os.environ, PYTHONPATH=str(ROOT / "upstream"))
    original = subprocess.run(
        [
            sys.executable,
            "-c",
            "from model.modeling_kgtreat import KGTREATForSequenceClassification",
        ],
        cwd=ROOT,
        env=original_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (ROOT / "run/original_import.log").write_text(
        original.stdout + original.stderr, encoding="utf-8"
    )
    sys.path.insert(0, str(compatible))
    from model.modeling_kgtreat import KGTREATForSequenceClassification
    from utils.configuration_utils import KGTREATConfig

    # Author's default sizes and the paper's time bins.
    config = KGTREATConfig(
        max_visit_time_embeddings=361, max_physical_time_embeddings=13
    )
    config._attn_implementation = "eager"
    # Artificial concept vectors stand in for the UMLS/BioLinkBERT file.
    concepts = np.random.normal(0, 0.1, (32, 1024)).astype("float32")
    np.save(ROOT / "run/artificial_concepts.npy", concepts)
    model = KGTREATForSequenceClassification(
        config, str(ROOT / "run/artificial_concepts.npy")
    )
    concept_projections = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear)
        and layer.in_features == 1024
        and layer.out_features == 200
    ]
    assert len(concept_projections) == 1, concept_projections
    history = torch.randint(1, 32, (4, 6)).repeat_interleave(2, dim=0)
    inputs = dict(
        input_ids=history,
        attention_mask=torch.ones(8, 6, dtype=torch.long),
        token_type_ids=torch.tensor([0, 1, 0, 1, 0, 1]).repeat(8, 1),
        visit_time_ids=torch.tensor([0, 30, 60, 90, 120, 150]).repeat(8, 1),
        physical_time_ids=torch.arange(6).repeat(8, 1),
        concept_ids=torch.arange(1, 6).repeat(8, 1),
        node_type_ids=torch.tensor([1, 0, 2, 3, 0]).repeat(8, 1),
        adj_lengths=torch.full((8,), 5),
        special_nodes_mask=torch.tensor([True, False, False, False, False]).repeat(
            8, 1
        ),
        edge_index=[
            torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
            for _ in range(8)
        ],
        edge_type=[torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]) for _ in range(8)],
        treatment_labels=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
        outcome_labels=torch.tensor([0, 0, 1, 1, 1, 1, 0, 0]),
        sep_graph_labels=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    assert torch.equal(history[0::2], history[1::2])
    torch.save(inputs, ROOT / "run/artificial_inputs.pt")
    model.eval()
    initial = model(**copy.deepcopy(inputs))
    g0, g1 = torch.autograd.grad(initial.loss, initial.logits)
    treatment = inputs["treatment_labels"][0::2]
    nonfactual_gradient = float(
        g0[treatment == 1].abs().sum() + g1[treatment == 0].abs().sum()
    )
    initial_loss = initial.loss.item()
    del initial, g0, g1
    # Forward pass, loss and gradients come from the author's code.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    parameter_before = {k: v.detach().clone() for k, v in model.named_parameters()}
    losses = []
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**copy.deepcopy(inputs))
        assert torch.isfinite(output.loss), "Nonfinite author loss"
        output.loss.backward()
        optimizer.step()
        losses.append(output.loss.item())
        print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    changed = sum(
        not torch.equal(parameter_before[k], v) for k, v in model.named_parameters()
    )
    del parameter_before, optimizer, output
    model.eval()
    with torch.no_grad():
        final = model(**copy.deepcopy(inputs))
    assert torch.isfinite(final.logits_factual).all()
    checkpoint = ROOT / "run/model.pt"
    torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    with torch.no_grad():
        repeated = model(**copy.deepcopy(inputs))
    reload_equal = all(torch.equal(a, b) for a, b in zip(final.logits, repeated.logits))
    assert reload_equal
    result = dict(
        status="passed",
        scope="original author forward/loss/backward and two updates on artificial inputs",
        original_import_exit_code=original.returncode,
        checkpoint_reloaded_predictions_equal=reload_equal,
        seed=20261029,
        patients=4,
        graph_entries=8,
        parameters=sum(p.numel() for p in model.parameters()),
        default_hidden_size=config.hidden_size,
        default_layers=config.num_hidden_layers,
        default_graph_layers=config.num_gnn_layers,
        default_concept_dim=config.concept_dim,
        concept_input_dimension=1024,
        concept_projection_modules=concept_projections,
        initial_loss=initial_loss,
        training_losses=losses,
        final_loss=final.loss.item(),
        nonfactual_branch_gradient_abs_sum=nonfactual_gradient,
        changed_parameter_tensors=changed,
        final_probabilities_y0=final.logits[0].softmax(-1).tolist(),
        final_probabilities_y1=final.logits[1].softmax(-1).tolist(),
        artificial_concepts_sha256=hashlib.sha256(
            (ROOT / "run/artificial_concepts.npy").read_bytes()
        ).hexdigest(),
        python=platform.python_version(),
        torch=torch.__version__,
        platform=platform.platform(),
        elapsed_seconds=time.perf_counter() - start,
    )
    assert changed > 0
    (ROOT / "run/result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
