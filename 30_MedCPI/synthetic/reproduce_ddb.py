# Build inputs from the public DDB graph and generated visits.

from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
ROOT = Path(__file__).resolve().parent
from fetch_author import fetch_author

RELATION_NAMES = [
    "belongstothecategoryof",
    "isacategory",
    "maycause",
    "isasubtypeof",
    "isariskfactorof",
    "isassociatedwith",
    "maycontraindicate",
    "interactswith",
    "belongstothedrugfamilyof",
    "child-parent",
    "isavectorfor",
    "mabeallelicwith",
    "seealso",
    "isaningradientof",
    "mabeindicatedby",
]
RAW_TO_DDB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "6": 5,
    "10": 6,
    "12": 7,
    "16": 8,
    "17": 9,
    "18": 10,
    "20": 11,
    "26": 12,
    "30": 13,
    "233": 14,
}
SEED_NAMES = [
    "Bronchial asthma",
    "Diabetes mellitus type 1",
    "Diabetes mellitus type 2",
    "Essential hypertension",
    "Hypertension, systemic",
    "Pneumonia",
    "Renal failure",
    "Renal failure, acute",
    "Renal failure, chronic",
    "Congestive heart failure",
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def download(item, path, drive_id=None):
    sys.path.insert(
        0,
        str(
            next(
                parent / "shared"
                for parent in ROOT.parents
                if (parent / "shared/downloads.py").is_file()
            )
        ),
    )
    from downloads import download_file

    url = (
        (
            "https://drive.usercontent.google.com/download?id="
            + drive_id
            + "&export=download&confirm=t"
        )
        if drive_id
        else item["url"]
    )
    download_file(url, path, item["sha256"], item.get("bytes"))


def main():
    manifest = json.loads(
        (ROOT / "public_inputs_manifest.json").read_text(encoding="utf-8")
    )
    author = fetch_author()
    sys.path.insert(0, str(author / "scripts"))
    assets = ROOT / "assets"
    archive = assets / "data_preprocessed_biomed.zip"
    download(manifest["knowledge"], archive, manifest["knowledge"]["drive_id"])
    for item in manifest["sapbert"]["files"]:
        download(item, assets / "sapbert" / item["path"])
    with zipfile.ZipFile(archive) as z:
        content = {}
        for item in manifest["knowledge"]["members"]:
            payload = z.read(item["path"])
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError("DDB member checksum mismatch: " + item["path"])
            content[Path(item["path"]).name] = payload.decode("utf-8")
    names = json.loads(content["ddb_names.json"])
    raw_edges = json.loads(content["ddb_relas.json"])
    preferred = {info[0]: text for text, info in names.items() if info[1] == "1"}
    synonyms = defaultdict(list)
    for text, info in names.items():
        synonyms[info[0]].append(text)
    triples = sorted(
        set((row[0], row[1], RAW_TO_DDB[row[2]]) for row in raw_edges.values())
    )
    neighbors = defaultdict(set)
    for a, b, r in triples:
        if a in preferred and b in preferred:
            neighbors[a].add(b)
            neighbors[b].add(a)
    seed_ids = [
        names[name][0]
        for name in SEED_NAMES
        if name in names and names[name][0] in neighbors
    ]
    assert len(seed_ids) >= 6
    selected = list(dict.fromkeys(seed_ids))
    seen = set(selected)
    frontier = deque(selected)
    while frontier and len(selected) < 96:
        current = frontier.popleft()
        for node in sorted(neighbors[current]):
            if node not in seen:
                seen.add(node)
                selected.append(node)
                frontier.append(node)
                if len(selected) == 96:
                    break
    selected_set = set(selected)
    selected_edges = [
        (a, b, r) for a, b, r in triples if a in selected_set and b in selected_set
    ]
    assert selected_edges
    concept_ids = ["DDB:" + node for node in selected]
    concept_index = {node: i for i, node in enumerate(selected)}
    terms = [preferred[node] for node in selected]
    queries = []
    for node in seed_ids:
        alternatives = sorted(
            (s for s in synonyms[node] if s != preferred[node]),
            key=lambda s: (len(s), s),
        )
        queries.append(alternatives[0] if alternatives else preferred[node])

    print("Encoding real DDB concept names with the original SapBERT", flush=True)
    import torch
    from transformers import AutoModel, AutoTokenizer
    from sapbert_map_mentions import encode_texts, normalize_text

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(assets / "sapbert", local_files_only=True)
    encoder = AutoModel.from_pretrained(assets / "sapbert", local_files_only=True)
    encoder.eval()
    candidate_vectors = encode_texts(terms, tokenizer, encoder, "cpu", 16)
    query_vectors = encode_texts(queries, tokenizer, encoder, "cpu", 16)
    scores = query_vectors @ candidate_vectors.T
    best = scores.argmax(dim=1).tolist()
    exact_names = defaultdict(set)
    for text, info in names.items():
        exact_names[normalize_text(text)].add(info[0])
    mapping = []
    for i, j in enumerate(best):
        exact_candidates = sorted(exact_names[normalize_text(queries[i])])
        confirmed = exact_candidates[0] if len(exact_candidates) == 1 else None
        mapping.append(
            dict(
                public_mention=queries[i],
                source_ddb_id=seed_ids[i],
                nearest_candidate_ddb_id=selected[j],
                nearest_candidate_name=terms[j],
                cosine=float(scores[i, j]),
                exact_candidate_ids=exact_candidates,
                confirmed_exact_ddb_id=confirmed,
                confirmation=(
                    "unique_exact_public_alias" if confirmed else "unconfirmed"
                ),
            )
        )
    assert candidate_vectors.shape == (len(selected), 768)
    del encoder, tokenizer, candidate_vectors, query_vectors, scores

    cui_map = defaultdict(set)
    for line in content["ddb_to_umls_cui.txt"].splitlines():
        columns = line.split("\t")
        if (
            len(columns) >= 3
            and columns[2].startswith("C")
            and columns[1] in selected_set
        ):
            cui_map[columns[1]].add(columns[2])
    history_concepts = ["DDB:" + node for node in seed_ids]
    out = ROOT / "outputs/public_ddb"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for patient in range(48):
        visits = 2 + patient % 3
        nodes = [{"node_kind": "patient"}] + [
            {"node_kind": "visit"} for _ in range(visits)
        ]
        nodes += [dict(node_kind="concept", concept_id=c) for c in concept_ids]
        edges, sequence = [], []

        def edge(a, b, r):
            edges.append(dict(src_index=a, dst_index=b, relation_id=r))

        offset = 1 + visits
        for a, b, rel in selected_edges:
            edge(offset + concept_index[a], offset + concept_index[b], 4 + rel)
            edge(offset + concept_index[b], offset + concept_index[a], 19 + rel)
        for visit in range(visits):
            ids = [
                history_concepts[(patient + visit * 3 + j) % len(history_concepts)]
                for j in range(3)
            ]
            date = datetime(2000, 1, 1) + timedelta(days=patient * 3 + visit * 60)
            sequence.append(
                dict(
                    admittime=date.strftime("%Y-%m-%d %H:%M:%S"),
                    concepts=[dict(concept_id=c, source_type="diagnosis") for c in ids],
                )
            )
            edge(0, 1 + visit, 0)
            edge(1 + visit, 0, 1)
            for c in ids:
                j = concept_ids.index(c)
                edge(1 + visit, offset + j, 2)
                edge(offset + j, 1 + visit, 3)
        records.append(
            dict(
                instance_id="SYNTHETIC_DDB_%03d" % patient,
                label=patient % 2,
                ehr_sequence=sequence,
                pkg_nodes=nodes,
                pkg_edges=edges,
            )
        )
    input_path = out / "samples.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    details = dict(
        patient_records="entirely synthetic",
        knowledge_source="GreaseLM public Disease Database and DrugBank graph",
        concept_namespace="DDB concept identifiers; UMLS links are recorded separately",
        raw_relation_records=len(raw_edges),
        raw_unique_triples=len(triples),
        selected_concepts=len(selected),
        selected_real_direct_relations=len(selected_edges),
        real_relations_with_author_inverse_edges=2 * len(selected_edges),
        relation_names=RELATION_NAMES,
        relation_offset=4,
        inverse_offset=19,
        structural_relations={
            "0": "patient_to_visit",
            "1": "visit_to_patient",
            "2": "visit_to_concept",
            "3": "concept_to_visit",
        },
        umls_mappings={
            node: sorted(cui_map[node]) for node in selected if cui_map[node]
        },
        source_name_flag="0/1 marks whether the name is preferred",
        encoder="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        encoder_revision=manifest["sapbert"]["revision"],
        encoding_function="author sapbert_map_mentions.encode_texts",
        encoder_output_dimension=768,
        alignment_runner="Repository helper matches normalized DDB names and aliases; SapBERT nearest candidates are also recorded.",
        simulated_history_concept_assignment="Histories use DDB identifiers selected during data generation.",
        nearest_matches_same_as_source=sum(
            row["nearest_candidate_ddb_id"] == row["source_ddb_id"] for row in mapping
        ),
        exact_confirmations_same_as_source=sum(
            row["confirmed_exact_ddb_id"] == row["source_ddb_id"] for row in mapping
        ),
        mention_count=len(mapping),
        mapping=mapping,
        input_sha256=sha(input_path),
        source_manifest_sha256=sha(ROOT / "public_inputs_manifest.json"),
    )
    (out / "input_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    print(
        "Starting original Integrate network with the public DDB graph and synthetic patient visits",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "reproduce.py"),
            "--input-records",
            str(input_path),
            "--relation-count",
            "34",
            "--output-name",
            "outputs/public_ddb",
        ],
        check=True,
        cwd=ROOT,
    )
    print(
        json.dumps(
            dict(
                status="passed",
                concepts=len(selected),
                real_direct_edges=len(selected_edges),
                patients="synthetic",
                model="author Integrate",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
