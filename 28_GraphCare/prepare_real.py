# Read MIMIC tables and build inputs for the GraphCare task.

import ast
import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def raw_table(root, name):
    for p in [
        root / "hosp" / f"{name}.csv" / f"{name}.csv",
        root / "hosp" / f"{name}.csv",
        root / "hosp" / f"{name}.csv.gz",
    ]:
        if p.is_file():
            return p
    raise FileNotFoundError("Required original table missing: " + name)


def original_function(path, name, namespace):
    source = Path(path).read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return (
        namespace[name],
        hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
    )


def read_mapping(name, source, target, remove_dot=False, atc3=False):
    result = defaultdict(list)
    with (ROOT / "resources" / name).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code = row[source].strip()
            if remove_dot:
                code = code.replace(".", "")
            value = row[target].strip()
            if atc3:
                value = value[:4]
            result[code].append(value)
    return result


def prepare_mimic(mimic_root, count=512):
    import duckdb

    root = Path(mimic_root)
    private = ROOT / ".private"
    private.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.read_csv(str(raw_table(root, "admissions")), all_varchar=True).create_view(
        "admissions"
    )
    selected = con.execute(
        """SELECT subject_id FROM admissions GROUP BY subject_id
        HAVING count(*) BETWEEN 2 AND 4 AND count(admittime)=count(*) AND count(dischtime)=count(*)
        ORDER BY md5(subject_id || ':20261030:graphcare') LIMIT ?""",
        [count],
    ).fetchall()
    con.execute("CREATE TEMP TABLE selected(subject_id VARCHAR)")
    con.executemany("INSERT INTO selected VALUES (?)", selected)
    admissions = con.execute(
        """SELECT subject_id,hadm_id,admittime,dischtime,hospital_expire_flag
        FROM admissions JOIN selected USING(subject_id)
        ORDER BY subject_id,admittime,dischtime,hadm_id"""
    ).fetchall()
    spec = importlib.util.spec_from_file_location(
        "original_pyhealth_data", ROOT / "upstream_extra/pyhealth_data.py"
    )
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    patients = {}
    visits = {}
    for sid, hid, admission, discharge, flag in admissions:
        if sid not in patients:
            patients[sid] = data.Patient(
                patient_id=sid,
                birth_datetime=None,
                death_datetime=None,
                gender=None,
                ethnicity=None,
            )
        visit = data.Visit(
            visit_id=hid,
            patient_id=sid,
            encounter_time=datetime.fromisoformat(admission),
            discharge_time=datetime.fromisoformat(discharge),
            discharge_status=int(flag),
        )
        patients[sid].add_visit(visit)
        visits[(sid, hid)] = visit
    maps = {
        ("diagnoses_icd", "9"): read_mapping(
            "ICD9CM_to_CCSCM.csv", "ICD9CM", "CCSCM", True
        ),
        ("diagnoses_icd", "10"): read_mapping(
            "ICD10CM_to_CCSCM.csv", "ICD10CM", "CCSCM", True
        ),
        ("procedures_icd", "9"): read_mapping(
            "ICD9PROC_to_CCSPROC.csv", "ICD9PROC", "CCSPROC", True
        ),
        ("procedures_icd", "10"): read_mapping(
            "ICD10PROC_to_CCSPROC.csv", "ICD10PROC", "CCSPROC", True
        ),
        ("prescriptions", "NDC"): read_mapping(
            "NDC_to_ATC.csv", "NDC", "ATC", atc3=True
        ),
    }
    coverage = {}
    raw_visit_presence = defaultdict(Counter)
    mapped_visit_presence = defaultdict(Counter)
    for table in ["diagnoses_icd", "procedures_icd", "prescriptions"]:
        con.read_csv(str(raw_table(root, table)), all_varchar=True).create_view(table)
        source_rows = con.execute(
            f"SELECT count(*) FROM {table} JOIN selected USING(subject_id)"
        ).fetchone()[0]
        if table == "prescriptions":
            query = """SELECT subject_id,hadm_id,ndc,'NDC' FROM prescriptions JOIN selected USING(subject_id)
                WHERE hadm_id IS NOT NULL AND ndc IS NOT NULL ORDER BY subject_id,hadm_id,starttime,stoptime"""
        else:
            query = f"""SELECT subject_id,hadm_id,icd_code,icd_version FROM {table} JOIN selected USING(subject_id)
                WHERE hadm_id IS NOT NULL AND icd_code IS NOT NULL AND icd_version IS NOT NULL
                ORDER BY subject_id,hadm_id,CAST(seq_num AS INTEGER)"""
        rows = con.execute(query).fetchall()
        total = mapped = emitted = absent_visit = multiple_categories = 0
        missing_codes = set()
        input_codes = set()
        for sid, hid, raw, version in rows:
            if (sid, hid) not in visits:
                absent_visit += 1
                continue
            total += 1
            raw_visit_presence[(sid, hid)][table] += 1
            key = (
                raw.strip()
                if table == "prescriptions"
                else raw.strip().replace(".", "")
            )
            input_codes.add((version, key))
            targets = maps[(table, version)].get(key, [])
            if targets:
                mapped += 1
                mapped_visit_presence[(sid, hid)][table] += 1
                multiple_categories += int(len(set(targets)) > 1)
            else:
                missing_codes.add((version, key))
            for code in targets:
                visits[(sid, hid)].add_event(
                    data.Event(
                        code=code,
                        table=table,
                        vocabulary=(
                            "ATC"
                            if table == "prescriptions"
                            else ("CCSCM" if table == "diagnoses_icd" else "CCSPROC")
                        ),
                        visit_id=hid,
                        patient_id=sid,
                    )
                )
                emitted += 1
        coverage[table] = dict(
            all_selected_subject_source_rows=source_rows,
            dropped_missing_required_source_fields=source_rows - len(rows),
            eligible_nonmissing_rows=total,
            mapped_rows=mapped,
            unmapped_rows=total - mapped,
            unique_raw_codes=len(input_codes),
            unique_unmapped_codes=len(missing_codes),
            rows_with_multiple_distinct_target_categories=multiple_categories,
            mapped_event_rows=emitted,
            rows_without_selected_admission=absent_visit,
        )
    task, task_hash = original_function(
        ROOT / "upstream_extra/task_fn.py",
        "mortality_prediction_mimic4_fn",
        {"Patient": data.Patient, "Visit": data.Visit},
    )
    samples = []
    invalid_patients = 0
    for patient in patients.values():
        result = task(patient)
        invalid_patients += int(not result)
        samples.extend(result)
    assert samples
    # Check the author's next-visit labels against the admission order.
    labels_correct = True
    retained_visits = defaultdict(set)
    for sample in samples:
        retained_visits[sample["patient_id"]].add(sample["visit_id"])
    timing_overlap_samples = 0
    target_visit_in_input_samples = 0
    for sample in samples:
        patient = patients[sample["patient_id"]]
        index = next(
            i for i in range(len(patient)) if patient[i].visit_id == sample["visit_id"]
        )
        target = patient[index + 1]
        labels_correct &= sample["label"] == int(target.discharge_status)
        history = [
            patient[j]
            for j in range(index + 1)
            if patient[j].visit_id in retained_visits[patient.patient_id]
        ]
        timing_overlap_samples += int(
            any(v.discharge_time > target.encounter_time for v in history)
        )
        target_visit_in_input_samples += int(
            any(v.visit_id == target.visit_id for v in history)
        )
    assert labels_correct
    tables = ["diagnoses_icd", "procedures_icd", "prescriptions"]
    candidates = 0
    raw_complete = 0
    mapped_complete = 0
    absent_after_mapping = Counter()
    lost_all_to_mapping = Counter()
    for patient in patients.values():
        for i in range(len(patient) - 1):
            key = (patient.patient_id, patient[i].visit_id)
            candidates += 1
            raw_complete += int(all(raw_visit_presence[key][t] > 0 for t in tables))
            mapped_complete += int(
                all(mapped_visit_presence[key][t] > 0 for t in tables)
            )
            for table in tables:
                absent_after_mapping[table] += int(
                    mapped_visit_presence[key][table] == 0
                )
                lost_all_to_mapping[table] += int(
                    raw_visit_presence[key][table] > 0
                    and mapped_visit_presence[key][table] == 0
                )
    assert mapped_complete == len(samples)
    private_path = private / "real_task_samples.json"
    private_path.write_text(json.dumps(samples), encoding="utf-8")
    summary = dict(
        source=os.environ.get("KG_MODEL_DATASET", "Local MIMIC-IV hospital tables"),
        requested_patients=count,
        selected_patients=len(patients),
        complete_selected_admissions=len(admissions),
        patients_with_task_samples=len(patients) - invalid_patients,
        patients_without_task_samples=invalid_patients,
        task_samples=len(samples),
        positive_next_admission_labels=sum(s["label"] for s in samples),
        cohort_rule="Hash-order select patients with 2 to 4 complete admissions, before examining model output or labels; retain all admissions for task label chronology.",
        task="original mortality_prediction_mimic4_fn, unchanged function AST",
        task_function_sha256=task_hash,
        history_discharge_after_target_admission_samples=timing_overlap_samples,
        target_admission_in_input_samples=target_visit_in_input_samples,
        timing_policy="Preserve original author task even if admissions overlap; report overlap without relabeling or reselection.",
        complete_mimic_version=os.environ.get("KG_MODEL_MIMIC_VERSION", "local"),
        author_original_mimic_version="2.0",
        raw_to_category_coverage=coverage,
        next_admission_candidate_visits=candidates,
        candidate_visits_with_all_three_raw_event_types=raw_complete,
        candidate_visits_with_all_three_mapped_event_types=mapped_complete,
        otherwise_complete_candidate_visits_dropped_by_mapping=raw_complete
        - mapped_complete,
        candidate_visits_missing_after_mapping_by_table=dict(absent_after_mapping),
        candidate_visits_losing_all_codes_in_a_table_due_to_mapping=dict(
            lost_all_to_mapping
        ),
        unmapped_raw_codes="Original CrossMap behavior returns no mapped code; counts disclosed. No guessing or zero-filling mappings.",
        private_sample_file_sha256=digest(private_path),
    )
    con.close()
    return samples, summary


def prepare_knowledge(samples, umls_directory, neighbors_per_anchor=4):
    directory = Path(umls_directory)
    names = {}
    for line in (
        (directory / "concept_names.txt").read_text(encoding="utf-8").splitlines()
    ):
        cui, name = line.split("\t", 1)
        names[cui] = name
    ccs_file = ROOT / "resources/ccs_global_exact_matches.json"
    ccs_source = json.loads(ccs_file.read_text(encoding="utf-8"))
    ccs_matches = {(r["namespace"], r["code"]): r for r in ccs_source["records"]}
    category_names = {}
    for namespace in ["CCSCM", "CCSPROC", "ATC"]:
        for row in csv.DictReader(
            (ROOT / "resources" / f"{namespace}.csv").open(encoding="utf-8-sig")
        ):
            if namespace != "ATC" or row["level"] == "3.0":
                category_names[(namespace, row["code"])] = row["name"]
    atc = read_mapping("ATC_to_UMLS.csv", "ATC", "UMLS")
    used = defaultdict(Counter)
    for sample in samples:
        for field, namespace in [
            ("conditions", "CCSCM"),
            ("procedures", "CCSPROC"),
            ("drugs", "ATC"),
        ]:
            for visit in sample[field]:
                used[namespace].update(visit)
    code_to_node = {}
    node_names = {}
    mapping_records = []
    for namespace in ["CCSCM", "CCSPROC", "ATC"]:
        for code in sorted(used[namespace]):
            label = category_names.get((namespace, code))
            assert label, (namespace, code, "Missing published category name")
            if namespace == "ATC":
                candidates = set(atc.get(code, []))
            else:
                match = ccs_matches[(namespace, code)]
                assert match["name"] == label
                candidates = (
                    {x["cui"] for x in match["matches_in_full_public_kb"]}
                    if match["unique_global_match_in_graph"]
                    else set()
                )
            if len(candidates) == 1:
                node = next(iter(candidates))
                method = (
                    "author_ATC_to_UMLS"
                    if namespace == "ATC"
                    else "unique_exact_alias_in_full_public_scispacy_KB_and_author_graph"
                )
            else:
                node = namespace + ":" + code
                method = "unresolved_category_retained_without_invented_edges"
            code_to_node[(namespace, code)] = node
            node_names[node] = names.get(node, label)
            mapping_records.append(
                dict(
                    namespace=namespace,
                    code=code,
                    node=node,
                    method=method,
                    original_graph_has_name=node in names,
                    history_occurrences=used[namespace][code],
                )
            )
    anchors = set(code_to_node.values())
    incident = defaultdict(set)
    with (directory / "umls.csv").open(encoding="utf-8") as f:
        for line in f:
            relation, a, b, weight = line.rstrip("\n").split("\t")
            if a in anchors:
                incident[a].add((a, relation, b))
            if b in anchors:
                incident[b].add((a, relation, b))
    chosen = set()
    for anchor in sorted(anchors):
        chosen.update(sorted(incident[anchor])[:neighbors_per_anchor])
    graph_nodes = set(anchors)
    for a, rel, b in chosen:
        graph_nodes.update([a, b])
    for node in graph_nodes:
        if node not in node_names:
            node_names[node] = names[node]
    # k_hop_subgraph takes the node count from the largest edge index,
    # so number the isolated categories first.
    connected = {x for a, r, b in chosen for x in (a, b)}
    nodes = sorted(graph_nodes - connected) + sorted(connected)
    assert len(nodes) == len(graph_nodes) and nodes[-1] in connected
    relations = sorted({r for a, r, b in chosen})
    index = {x: i for i, x in enumerate(nodes)}
    assert relations
    category_maps = {
        namespace: {
            code: index[node]
            for (ns, code), node in code_to_node.items()
            if ns == namespace
        }
        for namespace in ["CCSCM", "CCSPROC", "ATC"]
    }
    per_type = {}
    for ns in ["CCSCM", "CCSPROC", "ATC"]:
        rows = [r for r in mapping_records if r["namespace"] == ns]
        per_type[ns] = dict(
            unique_categories=len(rows),
            uniquely_mapped_categories=sum(
                r["method"] != "unresolved_category_retained_without_invented_edges"
                for r in rows
            ),
            categories_with_public_edges=sum(bool(incident[r["node"]]) for r in rows),
            total_history_occurrences=sum(r["history_occurrences"] for r in rows),
            history_occurrences_with_public_edges=sum(
                r["history_occurrences"] for r in rows if incident[r["node"]]
            ),
        )
    graph = dict(
        nodes=[dict(cui=x, name=node_names[x]) for x in nodes],
        relations=relations,
        triples=sorted(chosen),
    )
    summary = dict(
        real_knowledge_source="GraphCare author public UMLS subset",
        nodes=len(nodes),
        real_source_triples=len(chosen),
        relations=len(relations),
        ehr_anchor_nodes=len(anchors),
        isolated_ehr_anchors=sum(not incident[n] for n in anchors),
        category_coverage=per_type,
        ccs_exact_mapping_file_sha256=digest(ccs_file),
        ccs_mapping_source=ccs_source["source"],
        full_public_knowledge_concepts_scanned=ccs_source[
            "full_public_concepts_scanned"
        ],
        ccs_match_uniqueness_limit=ccs_source["limits"],
        node_numbering="Isolated categories first, connected concepts last; all node-indexed arrays follow this order. This satisfies original PyG inferred-node-count assumption without adding edges.",
        selected_edges_per_anchor_cap=neighbors_per_anchor,
        selection="First sorted original incident triples per anchor; no invented edges.",
        missing_original_resources=[
            "private CCSCM-to-UMLS and CCSPROC-to-UMLS mapping pickles",
            "original OpenAI vectors",
            "original whole-graph clustering inputs",
        ],
        approximation="Unresolved real category nodes are retained by their actual category names with no added medical edges; graph subset and vectors differ from original pipeline.",
    )
    (ROOT / ".private/knowledge_input.json").write_text(
        json.dumps(dict(graph=graph, mapping=mapping_records), indent=2),
        encoding="utf-8",
    )
    return graph, category_maps, summary
