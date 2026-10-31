# Generate artificial graph tensors for GraphCare.

import numpy as np
import torch
from torch_geometric.data import Data


def generate(seed=20261029):
    rng = np.random.RandomState(seed)
    num_nodes, num_rels, max_visit, embedding_dim = 32, 5, 3, 1536
    node_embedding = torch.tensor(
        rng.normal(0, 0.1, (num_nodes, embedding_dim)), dtype=torch.float32
    )
    relation_embedding = torch.tensor(
        rng.normal(0, 0.1, (num_rels, embedding_dim)), dtype=torch.float32
    )
    samples = []
    for patient in range(256):
        nodes = rng.choice(num_nodes, 12, replace=False)
        edges = [(i, (i + step) % 12) for i in range(12) for step in [1, 2]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        relation = torch.tensor(rng.randint(0, num_rels, len(edges)), dtype=torch.long)
        visits = np.zeros((max_visit, num_nodes), dtype=np.float32)
        for visit in range(1 + patient % max_visit):
            visits[visit, rng.choice(nodes, 4, replace=False)] = 1
        ehr_nodes = (visits.sum(axis=0) > 0).astype(np.float32)
        graph = Data(
            edge_index=edge_index,
            y=torch.tensor(nodes, dtype=torch.long),
            relation=relation,
            visit_padded_node=torch.tensor(visits),
            ehr_nodes=torch.tensor(ehr_nodes),
            label=torch.tensor([patient % 2], dtype=torch.float32),
            num_nodes=len(nodes),
        )
        samples.append(graph)
    summary = {
        "kind": "entirely_simulated_graphs_visits_embeddings_and_binary_labels",
        "seed": seed,
        "patients": 256,
        "split": {"train": 128, "validation": 64, "test": 64},
        "vocabulary_nodes": num_nodes,
        "relation_types": num_rels,
        "nodes_per_patient_graph": 12,
        "edges_per_patient_graph": 24,
        "max_visits": max_visit,
        "concepts_per_visit": 4,
        "embedding_dimension": embedding_dim,
        "embedding_source": "Independent normal draws, mean 0 and standard deviation 0.1; no language model or medical knowledge is encoded.",
        "graph_generation": "Twelve uniformly sampled vocabulary nodes; directed edges to the next one and next two local positions; uniformly sampled relation indices.",
        "visit_generation": "One to three visits; each selects four nodes from that patient graph, then pads remaining visits with zeros.",
        "label_generation": "Alternating 0 and 1 labels independent of the simulated clinical features.",
        "clinical_meaning": "None. These artificial tensors exercise the original binary prediction model only.",
    }
    return samples, node_embedding, relation_embedding, summary
