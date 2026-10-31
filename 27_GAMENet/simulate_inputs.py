# Generate artificial visits in the author's input format.

import json
from pathlib import Path
import random
import dill
from read_legacy_vocabulary import convert


def generate(author, patients=36, seed=20261029):
    author = Path(author)
    data = author / "data"
    vocabulary, conversion = convert(
        Path(__file__).resolve().parent / "upstream/data/voc_final.pkl",
        data / "voc_final.pkl",
    )
    sizes = [
        len(vocabulary[name].idx2word) for name in ["diag_voc", "pro_voc", "med_voc"]
    ]
    rng = random.Random(seed)
    records = []
    for patient in range(patients):
        visits = []
        for visit in range(2 + patient % 3):
            diagnosis = sorted(rng.sample(range(sizes[0]), 4))
            procedure = sorted(rng.sample(range(sizes[1]), 2))
            medications = sorted(rng.sample(range(sizes[2]), 5))
            visits.append([diagnosis, procedure, medications])
        records.append(visits)
    dill.dump(records, (data / "records_final.pkl").open("wb"))
    split = int(patients * 2 / 3)
    test_count = int((patients - split) / 2)
    summary = {
        "kind": "entirely_simulated_patient_sequences",
        "seed": seed,
        "patients": patients,
        "visits": sum(map(len, records)),
        "visits_per_patient": [2, 3, 4],
        "codes_per_visit": {"diagnoses": 4, "procedures": 2, "medications": 5},
        "generation": "Uniform sampling without replacement from author vocabulary indices; The selected codes form artificial visits.",
        "vocabulary_sizes": sizes,
        "vocabulary_serialization_compatibility": conversion,
        "graphs": "Unchanged author-public EHR and DDI adjacency matrices.",
        "split": {
            "train_patients": split,
            "test_patients": test_count,
            "validation_patients": patients - split - test_count,
        },
        "optimizer_steps_per_epoch": sum(map(len, records[:split])),
    }
    (data / "simulation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
