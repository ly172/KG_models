from pathlib import Path
import sys, json, numpy as np, torch

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
sys.path.insert(0, str(p / "run/compatible"))
from model.modeling_kgtreat import KGTREATForSequenceClassification
from utils.configuration_utils import KGTREATConfig

torch.set_num_threads(2)
torch.manual_seed(20261029)
cfg = KGTREATConfig(max_visit_time_embeddings=361, max_physical_time_embeddings=13)
cfg._attn_implementation = "eager"
file = p / "run/artificial_concepts.npy"
m = KGTREATForSequenceClassification(cfg, str(file))
provided = torch.from_numpy(np.load(file))
actual = m.kgtreat.concept_emb.emb.weight[:32].detach()
result = {
    "source": "author constructor in locked environment",
    "provided_shape": list(provided.shape),
    "frozen": not m.kgtreat.concept_emb.emb.weight.requires_grad,
    "preserved_values": bool(torch.equal(provided, actual)),
    "max_abs_difference": float((provided - actual).abs().max()),
    "input_standard_deviation": float(provided.std()),
    "initialized_table_standard_deviation": float(actual.std()),
    "interpretation": "Original constructor post_init applies BERT embedding initialization, including the frozen concept lookup table. This package preserves this behavior and starts from random weights.",
}
(p / "run/initialization_probe.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
