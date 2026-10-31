# Select author graph records and generate artificial visits.

import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data


def select_graph(directory, maximum_nodes=32, seed_cui="C0011849"):
    directory = Path(directory)
    names = {}
    with (directory / "concept_names.txt").open(encoding="utf-8") as stream:
        for line in stream:
            cui, name = line.rstrip("\n").split("\t", 1)
            names[cui] = name
    if seed_cui not in names:
        raise RuntimeError("Preset seed concept is absent from author names")
    neighbors = set()
    source_rows = 0
    incident_rows = 0
    with (directory / "umls.csv").open(encoding="utf-8") as stream:
        for line in stream:
            relation, source, target, weight = line.rstrip("\n").split("\t")
            source_rows += 1
            if source == seed_cui or target == seed_cui:
                incident_rows += 1
                neighbors.add(target if source == seed_cui else source)
    neighbors.discard(seed_cui)
    nodes = [seed_cui] + sorted(neighbors)[: maximum_nodes - 1]
    selected = set(nodes)
    triples = set()
    selected_source_rows = 0
    with (directory / "umls.csv").open(encoding="utf-8") as stream:
        for line in stream:
            relation, source, target, weight = line.rstrip("\n").split("\t")
            if source in selected and target in selected:
                selected_source_rows += 1
                triples.add((source, relation, target))
    triples = sorted(triples)
    relations = sorted({relation for _, relation, _ in triples})
    assert (
        len(nodes) == maximum_nodes and triples and all(node in names for node in nodes)
    )
    graph = {
        "seed_cui": seed_cui,
        "seed_name": names[seed_cui],
        "nodes": [{"cui": node, "name": names[node]} for node in nodes],
        "relations": relations,
        "triples": triples,
        "selection": "Seed C0011849 fixed before model output; take the first 31 unique neighbor CUIs in lexical order, then retain all source triples whose endpoints are in the 32-node set.",
        "source_rows_scanned": source_rows,
        "seed_incident_rows": incident_rows,
        "seed_unique_neighbors": len(neighbors),
        "selected_source_rows": selected_source_rows,
        "unique_selected_triples": len(triples),
        "invented_or_reversed_edges": 0,
        "edge_weight_handling": "The fourth source column is not supplied to GraphCare; the author UMLS sampling script also represents a triple by source, relation, target only.",
    }
    return graph


def make_simulated_samples(graph, node_embedding, seed=20261029):
    nodes = [item["cui"] for item in graph["nodes"]]
    node_ids = {node: index for index, node in enumerate(nodes)}
    relation_ids = {
        relation: index for index, relation in enumerate(graph["relations"])
    }
    edges = [
        [node_ids[source], node_ids[target]] for source, _, target in graph["triples"]
    ]
    relations = [relation_ids[relation] for _, relation, _ in graph["triples"]]
    G = Data(
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        relation=torch.tensor(relations, dtype=torch.long),
        y=torch.arange(len(nodes)),
        x=node_embedding.clone(),
        num_nodes=len(nodes),
    )
    rng = np.random.RandomState(seed)
    samples = []
    for patient in range(640):
        visits = np.zeros((3, len(nodes)), dtype=np.float32)
        for visit in range(1 + patient % 3):
            visits[visit, 0] = 1
            visits[visit, rng.choice(np.arange(1, len(nodes)), 3, replace=False)] = 1
        direct = np.where(visits.sum(axis=0) > 0)[0]
        samples.append(
            {
                "node_set": direct.tolist(),
                "ehr_node_set": torch.tensor(
                    (visits.sum(axis=0) > 0).astype(np.float32)
                ),
                "visit_padded_node": torch.tensor(visits),
                "label": torch.tensor([patient % 2], dtype=torch.float32),
                "patient_id": "simulated_%04d" % patient,
            }
        )
    permutation = np.random.RandomState(528).permutation(len(samples))
    samples = [samples[index] for index in permutation]
    summary = {
        "kind": "synthetic_patient_visits_and_labels_on_real_public_knowledge_graph",
        "patients": 640,
        "train_patients": 512,
        "validation_patients": 64,
        "test_patients": 64,
        "max_visits": 3,
        "visit_generation": "Every simulated visit contains the fixed seed plus three uniformly sampled neighbors. Each patient has one to three visits.",
        "label_generation": "Alternating binary labels assigned before patient permutation, independent of clinical meaning.",
        "generation_seed": seed,
        "patient_permutation_seed": 528,
    }
    return G, [samples[:512], samples[512:576], samples[576:]], summary
