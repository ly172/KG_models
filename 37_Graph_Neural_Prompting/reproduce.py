from pathlib import Path
import sys, ast, types, json, math, copy, difflib, time
from prepare import prepare

ROOT = Path(__file__).resolve().parent
started = time.perf_counter()
prepare()
V = ROOT / "outputs"
V.mkdir(exist_ok=True)

import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from scipy.sparse import coo_matrix

torch.set_num_threads(4)
torch.manual_seed(12024)
np.random.seed(12024)
S = ROOT / "cache/author"
source = (S / "llm.py").read_text(encoding="utf-8")
gs = (S / "gnn.py").read_text(encoding="utf-8")


def chosen(s, names):
    return ast.Module(
        body=[
            n
            for n in ast.parse(s).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ],
        type_ignores=[],
    )


base = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "sys": sys}
gn = dict(base)
exec(
    compile(
        chosen(
            gs, {"OriginConceptEmbedding", "gelu", "GELU", "Decoder", "DistMultDecoder"}
        ),
        "author_gnn",
        "exec",
    ),
    gn,
)
gnn = types.SimpleNamespace(**gn)
args = types.SimpleNamespace(
    use_relational_gnn=False,
    gnn_layers=3,
    gnn_dim=1024,
    n_ntype=4,
    n_etype=38,
    freeze_ent_emb=True,
    p_dropout_gnn=0.2,
    cross_modality_layers=1,
    token_dim=2048,
    moe_experts=0,
    link_task=0.1,
    link_decoder="DistMult",
    link_proj_headtail=False,
    link_normalize_headtail=0,
    link_negative_adversarial_sampling=True,
    link_negative_adversarial_sampling_temperature=1.0,
    link_regularizer_weight=0.01,
    scaled_distmult=False,
    num_choice=4,
    no_projector=False,
    use_cross_modality_for_text=False,
)
names = {"GAT", "CrossModalityMHA", "MyGNN", "MyModel"}


def classes(s):
    ns = {
        **base,
        "args": args,
        "gnn": gnn,
        "GATConv": GATConv,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoTokenizer": AutoTokenizer,
    }
    exec(compile(chosen(s, names), "author_llm_components", "exec"), ns)
    return ns


cpu_source = source.replace(".to('cuda:0')", ".to('cpu')").replace(
    "max_memory=my_max_memory", "max_memory={'cpu':'16GiB'}"
)
cpu_source = cpu_source.replace(
    "torch_dtype=torch.bfloat16", "torch_dtype=torch.float32"
)
original = classes(cpu_source)
example = json.loads((ROOT / "data/example.json").read_text())
questions = [example["question"]]
graphs = []
for item in example["graphs"]:
    a = item["adj"]
    graphs.append(
        {
            "adj": coo_matrix(
                (np.array(a["data"], dtype=np.uint8), (a["row"], a["col"])),
                shape=tuple(a["shape"]),
            ),
            "concepts": np.array(item["concepts"], dtype=np.int64),
            "qmask": np.array(item["qmask"], dtype=bool),
            "amask": np.array(item["amask"], dtype=bool),
            "cid2score": {c: s for c, s in item["cid2score"]},
        }
    )
N = 200
used = example["embedding_concept_ids"]
cidmap = {c: i for i, c in enumerate(used)}
cp = torch.tensor(np.load(ROOT / "data/concept_embeddings.npy"), dtype=torch.float32)
args.concept_num = len(used)
args.concept_in_dim = 1024


def prepare(graphs):
    import pickle, itertools, os
    from tqdm import tqdm

    ds = (S / "utils/data_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(ds)
    ns = {
        "torch": torch,
        "np": np,
        "sys": sys,
        "os": os,
        "pickle": pickle,
        "itertools": itertools,
        "tqdm": tqdm,
    }
    for class_name, methods in [
        ("DataLoader", {"load_sparse_adj_data_with_contextnode"}),
        (
            "MultiGPUSparseAdjDataBatchGenerator",
            {"process_graph_data", "_process_one_graph"},
        ),
    ]:
        node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        node.body = [
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "author_data_utils", "exec"
            ),
            ns,
        )
    args.prompt_design = False
    args.load_graph_cache = False
    args.dump_graph_cache = False
    args.local_rank = -1
    args.kg_only_use_qa_nodes = False
    args.max_num_relation = -1
    args.link_drop_max_count = 100
    args.link_drop_probability = 0.3
    args.link_drop_probability_in_which_keep = 0.2
    args.link_negative_sample_size = 64
    loader = ns["DataLoader"]()
    loader.args = args
    loader.cxt_node_connects_all = False
    loader.num_choice = 4
    loader.id2relation = list(range(17))
    loader.debug = False
    graphfile = V / "original_graphs.pk"
    with graphfile.open("wb") as f:
        for graph in graphs:
            pickle.dump(graph, f)
    ids, nt, scores, lengths, special, (edges, relations) = (
        loader.load_sparse_adj_data_with_contextnode(
            str(graphfile), N, None, mode="train"
        )
    )
    original_ids = ids.clone()
    for cid in used:
        ids[original_ids == cid + 1] = cidmap[cid] + 1
    sampler = ns["MultiGPUSparseAdjDataBatchGenerator"]()
    sampler.args = args
    sampler.eval_end_task_mode = False
    edges, relations, pos, neg = sampler.process_graph_data(edges, relations, nt)
    return [edges, relations, pos, neg, ids, nt, scores, lengths]


inputs = prepare(graphs)

args.encoder = str(ROOT / "cache/flan-t5-xl")
args.max_grad_norm = 1.0
args.prompt = "gnn"
args.lora = False
args.use_wandb = False
args.dataset_level_prompt = False
model = original["MyModel"](args, cp)
model.llm.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)
texts = [
    "Question: "
    + q["question"]["stem"]
    + "\nOptions: "
    + ", ".join(c["text"] for c in q["question"]["choices"])
    for q in questions
]
token_counts = [len(tokenizer(text)["input_ids"]) for text in texts]
assert max(token_counts) <= 100
tok = tokenizer(
    texts, return_tensors="pt", padding="max_length", truncation=True, max_length=100
)
answers = [
    next(c["text"] for c in q["question"]["choices"] if c["label"] == q["answerKey"])
    for q in questions
]
labels = tokenizer(
    answers, return_tensors="pt", padding="max_length", truncation=True, max_length=100
)["input_ids"]
labels[labels == tokenizer.pad_token_id] = -100
optimizer = torch.optim.AdamW(model.gnn_model.parameters(), lr=1e-4)
before = {
    n: p.detach().clone()
    for n, p in model.gnn_model.named_parameters()
    if p.requires_grad
}
losses = []
for step in range(2):
    model.train()
    optimizer.zero_grad()
    with torch.autocast("cpu", enabled=False):
        outputs, link_loss, _ = model(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            labels=labels,
            graph_inputs=copy.deepcopy(inputs),
        )
        loss = outputs.loss + args.link_task * link_loss
    print("STEP", step, "LOSS", float(loss.detach()), flush=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    losses.append(float(loss.detach()))
changed = [
    n
    for n, p in model.gnn_model.named_parameters()
    if n in before and not torch.equal(before[n], p)
]
assert changed and all(math.isfinite(x) for x in losses)
model.eval()
with torch.no_grad(), torch.autocast("cpu", enabled=False):
    outputs, _, _ = model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        labels=labels,
        graph_inputs=copy.deepcopy(inputs),
    )
    logits = outputs.logits.detach().clone()
torch.save(model.gnn_model.state_dict(), V / "adapter.pt")
fresh_graph_network = original["MyGNN"](args, cp)
fresh_graph_network.load_state_dict(
    torch.load(V / "adapter.pt", map_location="cpu", weights_only=True), strict=True
)
model.gnn_model = fresh_graph_network
model.eval()
with torch.no_grad(), torch.autocast("cpu", enabled=False):
    outputs, _, _ = model(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        labels=labels,
        graph_inputs=copy.deepcopy(inputs),
    )
assert torch.equal(logits, outputs.logits)
with torch.no_grad():
    generated = model.generate(
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        graph_inputs=copy.deepcopy(inputs),
    )
generated_answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
assert all(answer.strip() for answer in generated_answers)
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in [
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            ]
        ]

    info = ProcessMemory()
    info.cb = ctypes.sizeof(info)
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    assert fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(info), info.cb)
    peak_memory = int(info.PeakWorkingSetSize)
else:
    import resource

    peak_memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
report = {
    "status": "passed",
    "train_question_ids": [q["id"] for q in questions],
    "optimizer_steps": 2,
    "losses": losses,
    "changed_parameter_tensors": len(changed),
    "llm": "google/flan-t5-xl",
    "model": "author MyModel plus MyGNN",
    "changes": ["CUDA placement uses CPU", "CPU memory map", "CPU float32 arithmetic"],
    "input_kind": "real OpenBookQA training question and author ConceptNet graphs",
    "question_token_counts": token_counts,
    "maximum_tokens": 100,
    "maximum_nodes": 200,
    "gnn_layers": 3,
    "gnn_dimension": 1024,
    "seconds": time.perf_counter() - started,
}
report.update(
    {
        "training_settings": {
            "scaled_distmult": args.scaled_distmult,
            "link_negative_adversarial_sampling": args.link_negative_adversarial_sampling,
            "p_dropout_gnn": args.p_dropout_gnn,
            "link_task": args.link_task,
            "seed": 12024,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "learning_rate_schedule": "constant",
            "max_grad_norm": args.max_grad_norm,
        },
        "generated_answers": generated_answers,
        "gold_answers": answers,
        "peak_working_set_bytes": peak_memory,
    }
)
(V / "cpu_precision.patch").write_text(
    "".join(
        difflib.unified_diff(
            source.splitlines(True),
            cpu_source.splitlines(True),
            fromfile="author/llm.py",
            tofile="runtime/llm_cpu.py",
        )
    ),
    encoding="utf-8",
)
(V / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
