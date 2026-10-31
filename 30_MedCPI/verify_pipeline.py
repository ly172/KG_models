# Summarize the run and check that train and test patients do not overlap.

from collections import Counter
import hashlib
from datetime import datetime
import json
from pathlib import Path
from prepare_public_inputs import lines

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads(
        (ROOT / ".private/outputs" / relative).read_text(encoding="utf-8")
    )


def main():
    aligned = read("aligned_ehr/mimic4/aligned_ehr_stats.json")
    construct = read("construct/mimic4/mortality/construct_stats.json")
    personalize = read("personalize/mimic4/mortality/pkg_stats.json")
    subjects = {
        split: {
            str(r["patient_id"])
            for r in lines(
                ROOT / f".private/outputs/splits/mimic4/mortality/{split}.jsonl"
            )
        }
        for split in ["train", "valid", "test"]
    }
    disjoint = not any(
        subjects[a] & subjects[b]
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    )
    assert disjoint and construct["num_edges"] > 0
    kg_counts = Counter()
    unmapped_as_anchor = 0
    for patient in lines(
        ROOT / ".private/outputs/aligned_ehr/mimic4/patients_visits_aligned.jsonl"
    ):
        for visit in patient["visits"]:
            for event in visit["aligned_events"]:
                kind = event["source_type"]
                kg_counts[kind + "_events"] += 1
                if event.get("kg_anchor"):
                    kg_counts[kind + "_umls_anchor_events"] += 1
                if event.get("concept_space") == "rxnorm":
                    kg_counts[kind + "_rxnorm_only_events"] += 1
                if not event.get("ehr_concept_id"):
                    kg_counts[kind + "_unmapped_events"] += 1
                if event.get("alignment_status") == "unmapped" and event.get(
                    "kg_anchor"
                ):
                    unmapped_as_anchor += 1
    assert unmapped_as_anchor == 0
    visits = {
        str(patient["patient_id"]): {str(v["visit_id"]): v for v in patient["visits"]}
        for patient in lines(
            ROOT / ".private/outputs/preprocessed/mimic4/patients_visits.jsonl"
        )
    }
    overlap_instances = 0
    target_in_history = 0
    instances = 0
    for split in ["train", "valid", "test"]:
        for row in lines(
            ROOT / f".private/outputs/splits/mimic4/mortality/{split}.jsonl"
        ):
            instances += 1
            context = [str(v) for v in row["context_visit_ids"]]
            target_in_history += int(str(row["target_visit_id"]) in context)
            prediction = datetime.fromisoformat(row["prediction_time"])
            overlap_instances += int(
                any(
                    datetime.fromisoformat(
                        visits[str(row["patient_id"])][v]["dischtime"]
                    )
                    > prediction
                    for v in context
                )
            )
    assert target_in_history == 0
    source_manifest = json.loads(
        (ROOT / "author_source.json").read_text(encoding="utf-8")
    )
    result = {
        "real_source": json.loads(
            (ROOT / "verification/cohort.json").read_text(encoding="utf-8")
        )["source"],
        "synthetic_patient_rows": 0,
        "patient_splits_disjoint": disjoint,
        "aligned_event_counts": aligned["counts"],
        "knowledge_anchor_coverage": dict(kg_counts),
        "unmapped_events_used_as_kg_anchors": unmapped_as_anchor,
        "temporal_check": {
            "instances": instances,
            "target_visit_in_history": target_in_history,
            "history_discharge_after_prediction_instances": overlap_instances,
            "interpretation": "Author history is ordered by admission. Overlapping admissions can contain events after the target admission begins.",
        },
        "construct": {k: v for k, v in construct.items() if not k.endswith("_path")},
        "personalize": personalize,
        "scope": "Author model on a small MIMIC cohort with the documented public terminology subset.",
        "mapping_policy": "Author mapper outputs feed graph construction; comparisons with public sources are saved alongside the mappings.",
        "api_use": "Uses a GPT-5 relation schema published by the authors, plus local public terminology files.",
    }
    (ROOT / "verification/pipeline.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
