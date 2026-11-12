"""Freeze public BizBench enrollment before generation, within a Slurm CPU allocation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, body):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, sort_keys=True, allow_nan=False)


def episode_plan(selected, protocol):
    """Enumerate every intended cell before model generation; no result-dependent cells."""
    experiment = protocol["external"]["experiment"]
    core = set(protocol["models"]["core_ids"])
    return [dict(track="E", stage="core" if model["id"] in core else "extension",
                 model_id=model["id"], model=model["model"], model_revision=model["revision"],
                 task_id=task["task_id"], source_task=task["source_task"],
                 row_index=task["row_index"], public_sha256=task["public_sha256"],
                 scoring_policy_sha256=task["scoring_policy_sha256"],
                 method="external_code", library=library, seed=seed)
            for model in protocol["models"]["twenty_hf_models"]
            for task in selected for library in experiment["conditions"]
            for seed in experiment["seeds"]]


def main(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("enrollment requires Slurm; no model is called")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import pandas as pd
    from transformers import AutoConfig
    # The immutable deployment puts this helper beside the entry point. A local
    # import remains available for synthetic unit tests, never local data intake.
    try:
        import external_enrollment as rules
    except ModuleNotFoundError as exc:
        if exc.name != "external_enrollment":
            raise
        from benchmarks.agent_study import external_enrollment as rules
    from benchmarks.agent_study.external_packets import ScalarPolicy, packets, prompt
    from benchmarks.agent_study.external_study import interface_text
    from benchmarks.agent_study.rsi_qualify import verify_source
    from benchmarks.agent_study.transformers_chat import load_tokenizer, tokenize_chat, context_limit
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    source_hash = verify_source(args.qualified / "source")
    frozen = dict(protocol_sha256=sha(args.protocol), qualified_source_manifest_sha256=source_hash,
        helper_sha256={str(p.name): sha(p) for p in (Path(__file__), Path(rules.__file__))},
        slurm_job_id=os.environ["SLURM_JOB_ID"], started_utc=datetime.now(timezone.utc).isoformat(),
        scope="R1 deterministic enrollment only; no model or reference program execution")
    write(args.output / "plan.json", frozen)
    try:
        intake = json.loads((args.inputs / "terminal.json").read_text())
        if intake["status"] != "verified_public_sources":
            raise ValueError("verified source intake required")
        for item in intake["files"]:
            path = args.inputs / item["file"]
            if path.parent != args.inputs or sha(path) != item["sha256"]:
                raise ValueError("source intake bytes changed")
        if sha(args.train) != "d0af1868ac46b082f153981b813b97b78707153b7a5cd872b30490bec47b3cc1":
            raise ValueError("training input differs from the frozen release")
        sources = []
        for item in intake["files"]:
            name = item["file"]
            if name.endswith(".json"):
                source = "CodeFinQA" if name.startswith("czyssrs--") else "CodeTAT-QA"
                sources.append((source, item["sha256"], json.loads((args.inputs / name).read_text())))
        origins = rules.origin_index(sources)
        del sources
        train = pd.read_parquet(args.train)
        test = pd.read_parquet(args.inputs / "bizbench-test.parquet")
        included = {"CodeFinQA", "CodeTAT-QA", "CodeTATQA"}
        seen_train, train_templates, train_contexts, train_issuers = set(), set(), set(), set()
        invalid_training_contexts = 0
        for _, row in train[train.task.isin(included)].iterrows():
            try:
                key = rules.context_key(row.context, row.context_type)
                seen_train.add((rules.normalize(row.question), key))
                train_contexts.add(key)
                train_templates.add(rules.template_key(row.question))
                source = "CodeTAT-QA" if row.task == "CodeTATQA" else row.task
                issuer = rules.map_origin(source, row.question, key, origins)["issuer"]
                if issuer:
                    train_issuers.add(issuer)
            except (ValueError, TypeError):
                invalid_training_contexts += 1
                continue  # Invalid training contexts cannot establish an exact-input duplicate.
        policy = ScalarPolicy(protocol["external"]["unit_policy"]["candidate_instruction"],
            0, 0, "BizBench-derived strict numerical label match; original public units")
        interface = interface_text()
        write(args.output / "interface.json", dict(text=interface, sha256=hashlib.sha256(interface.encode()).hexdigest()))
        revision = protocol["external"]["revision"]
        ledger, eligible = [], []
        for index, row in test.iterrows():
            source = "CodeTAT-QA" if row.task == "CodeTATQA" else row.task
            item = dict(row_index=int(index), source_task=source, status="excluded", reason="source_outside_design")
            if row.task in included:
                try:
                    if not isinstance(row.context, str) or not row.context.strip():
                        raise ValueError("missing_context")
                    key = rules.context_key(row.context, row.context_type)
                    public, _ = packets(row.to_dict(), revision=revision, split="test", row_index=int(index),
                                        source_cluster="pending-enrollment", policy=policy)
                    if (rules.normalize(row.question), key) in seen_train:
                        raise ValueError("exact_normalized_train_overlap")
                    origin = rules.map_origin(source, row.question, key, origins)
                    template = rules.template_key(row.question)
                    item.update(status="eligible", reason=None, context_sha256=key, origin=origin,
                        sampling_group=origin["sampling_group"], dependency_block=origin["dependency_block"],
                        stable_key=f"{revision}/test/{index}", template_sha256=template,
                        development_exposure=dict(exact_input=False, same_context=key in train_contexts,
                            same_template=template in train_templates,
                            same_verified_issuer=origin["issuer"] in train_issuers),
                        units=rules.unit_tags(row.question, row.context), prompt_tokens={})
                    eligible.append((item, public))
                except (ValueError, TypeError) as exc:
                    item["reason"] = str(exc)[:200]
            ledger.append(item)
        # Tokenize the exact two condition prompts under every frozen core model.
        inventory = json.loads(args.inventory.read_text())
        models = {row["id"]: row for row in protocol["models"]["twenty_hf_models"]}
        tokenizer_receipts = []
        for model_id in protocol["models"]["core_ids"]:
            entries = [row for row in inventory if row["id"] == model_id and row.get("cached")]
            if len(entries) != 1 or any(entries[0][k] != models[model_id][k] for k in ("model", "revision")):
                raise ValueError("one matching cached core tokenizer entry required")
            row = entries[0]
            kwargs = dict(cache_dir=row["cache_check"]["cache"], local_files_only=True)
            tokenizer = load_tokenizer(row["model"], row["revision"], **kwargs)
            config = AutoConfig.from_pretrained(row["model"], revision=row["revision"], **kwargs)
            if context_limit(config, tokenizer) < 32768:
                raise ValueError("core model does not support the frozen common context")
            for item, public in eligible:
                counts = [len(tokenize_chat(tokenizer, prompt(public, interface_text=text))["input_ids"])
                          for text in ("", interface)]
                item["prompt_tokens"][model_id] = dict(scratch=counts[0], library=counts[1])
                if max(counts) + 2048 > 32768:
                    item.update(status="excluded", reason="common_context_contract_exceeded")
            tokenizer_receipts.append(dict(id=model_id, model=row["model"], revision=row["revision"],
                chat_template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()))
            del tokenizer, config
        enrollment_keys = [{key:item[key] for key in ("row_index", "source_task", "sampling_group", "stable_key")}
                           for item, _ in eligible if item["status"] == "eligible"]
        chosen = rules.select(enrollment_keys, cap=protocol["external"]["selection"]["per_source_groups_cap"])
        universe = defaultdict(list)
        for item in enrollment_keys:
            universe[(item["source_task"], item["sampling_group"])].append(item["row_index"])
        write(args.output / "group-universe.json", [dict(source_task=source, sampling_group=group,
            row_indices=sorted(indices), selection_sha256=rules.rank("group", source, group))
            for (source, group), indices in sorted(universe.items())])
        selected_ids = {item["row_index"] for item in chosen}
        selected = [item for item in ledger if item["row_index"] in selected_ids]
        # Known cross-source identical contexts keep one dependency block.
        parent = {item["dependency_block"]: item["dependency_block"] for item in selected}
        def find(key):
            while parent[key] != key:
                key = parent[key]
            return key
        def union(a, b):
            a, b = find(a), find(b)
            target = "unresolved" if "unresolved" in (a, b) else min(a, b)
            parent[a] = parent[b] = target
        contexts = defaultdict(list)
        for item in selected:
            contexts[item["context_sha256"]].append(item["dependency_block"])
        for blocks in contexts.values():
            for other in blocks[1:]:
                union(blocks[0], other)
        counts = Counter(item["source_task"] for item in selected)
        if set(counts) != {"CodeFinQA", "CodeTAT-QA"}:
            raise ValueError("both prespecified external populations must be nonempty")
        weights = defaultdict(float)
        with (args.output / "public.jsonl").open("x", encoding="utf-8") as pub, (args.output / "scoring.jsonl").open("x", encoding="utf-8") as scoring:
            for item in selected:
                item["dependency_block"] = find(item["dependency_block"])
                weights[item["dependency_block"]] += .5 / counts[item["source_task"]]
                row = test.loc[item["row_index"]].to_dict()
                public, private = packets(row, revision=revision, split="test", row_index=item["row_index"],
                    source_cluster=item["dependency_block"], policy=policy)
                item["public_sha256"] = private["public_sha256"]
                item["scoring_policy_sha256"] = private["scoring_policy_sha256"]
                item["task_id"] = public["id"]
                pub.write(json.dumps(public, sort_keys=True) + "\n")
                scoring.write(json.dumps(private, sort_keys=True) + "\n")
        write(args.output / "eligibility-ledger.json", ledger)
        write(args.output / "selected-metadata.json", selected)
        write(args.output / "tokenizers.json", tokenizer_receipts)
        cells = episode_plan(selected, protocol)
        write(args.output / "episode-plan.json", cells)
        effective = 1 / sum(value * value for value in weights.values())
        result = dict(status="enrollment_frozen", model_calls=0, source_counts=dict(Counter(test.task)),
            eligible_counts=dict(Counter(item["source_task"] for item in ledger if item["status"] == "eligible")),
            exclusion_counts=dict(Counter(item["reason"] for item in ledger if item["status"] == "excluded")),
            selected_counts=dict(counts), selected_questions=len(selected), dependency_blocks=len(weights),
            available_groups=dict(Counter(source for source, _ in universe)),
            episode_counts=dict(Counter(cell["stage"] for cell in cells)),
            train_exposure=dict(invalid_contexts=invalid_training_contexts,
                compared_inputs=len(seen_train), verified_issuers=len(train_issuers)),
            effective_equal_weight_groups=effective, optimistic_normal_plan_halfwidth=1.959963984540054/math.sqrt(effective),
            unresolved_weight=weights.get("unresolved", 0), independent_population_inference=False,
            interpretation="Fixed public benchmark; unresolved origins do not create independent documents",
            protocol_sha256=frozen["protocol_sha256"], qualified_source_manifest_sha256=source_hash,
            files={name:sha(args.output / name) for name in ("public.jsonl", "scoring.jsonl", "eligibility-ledger.json",
                   "selected-metadata.json", "tokenizers.json", "interface.json", "episode-plan.json",
                   "group-universe.json")}, publication_ready=False)
        if (verify_source(args.qualified / "source") != source_hash or sha(args.protocol) != frozen["protocol_sha256"]
                or any(sha(p) != frozen["helper_sha256"][p.name] for p in (Path(__file__), Path(rules.__file__)))):
            raise ValueError("source or protocol changed during enrollment")
    except BaseException as exc:
        write(args.output / "terminal.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}",
              model_calls=0, publication_ready=False))
        raise
    write(args.output / "terminal.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "train", "qualified", "protocol", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
