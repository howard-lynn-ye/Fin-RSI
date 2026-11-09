"""Complete seven-method planning and accounting; no inference or implicit retries."""
from collections import Counter, defaultdict
import json
from pathlib import Path

METHOD_NAMES = ("no_improve", "self_debug", "check_feedback", "reflexion",
                "playbook", "team", "evolve")
LEARNERS = ("playbook", "evolve")


def partial_accounting(path):
    """Terminal evidence after an interrupt; unknown or absent usage remains unknown."""
    path = Path(path)
    if not path.exists():
        return dict(available=False, started_requests=None, known_generated_tokens=None,
                    unknown_cost_requests=None, reason="provider ledger was not created")
    started = finished = known = tokens = 0
    incomplete = False
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            incomplete = True
            break
        event = json.loads(line)
        if event.get("backend_mode") != "bounded":
            continue
        if event["kind"] == "call_started":
            started += 1
        elif event["kind"] in ("call_completed", "call_failed"):
            finished += 1
            cost = event.get("tokens") if event["kind"] == "call_completed" else event.get("token_cost")
            if cost is not None:
                if type(cost) is not int or cost < 0:
                    raise ValueError("invalid known provider cost")
                known += 1
                tokens += cost
    if finished > started or known > started:
        raise ValueError("provider outcomes exceed started requests")
    return dict(available=True, started_requests=started, known_generated_tokens=tokens,
        unknown_cost_requests=started-known, unfinished_requests=started-finished,
        incomplete_final_event=incomplete)


def build_units(protocol, tasks):
    """One independent fit and all test tasks/seeds belong to the same execution unit."""
    spec = protocol["seven_methods"]
    if (spec["names"] != list(METHOD_NAMES) or spec["learning_methods"] != list(LEARNERS)
            or spec["models"] != protocol["models"]["core_ids"]
            or spec["library_conditions"] != [False, True]
            or spec["evaluation_seeds"] != [101, 211, 307] or spec["fit_seed"] != 11):
        raise ValueError("seven-method design differs from the frozen protocol")
    if ([t.manifest() for t in tasks.dev] != protocol["positions"]["dev_tasks"]
            or [t.manifest() for t in tasks.heldout] != protocol["positions"]["evaluation_tasks"]):
        raise ValueError("development/evaluation task manifests differ")
    if {t.algorithm for t in tasks.dev} & {t.algorithm for t in tasks.heldout}:
        raise ValueError("fit and test algorithms overlap")
    frozen = spec["frozen_method_parameters"]
    if [t.id for t in sorted(tasks.dev, key=lambda t:t.id)[:4]] != frozen["Evolve"]["actual_fit_batch_ids"]:
        raise ValueError("Evolve development minibatch differs")
    validation = set(sorted({t.algorithm for t in tasks.dev})[1::2])
    split = {"development_ids":[t.id for t in tasks.dev if t.algorithm not in validation],
             "validation_ids":[t.id for t in tasks.dev if t.algorithm in validation]}
    if any(split[k] != frozen["PlaybookLearning"][k] for k in split):
        raise ValueError("Playbook development/validation split differs")
    models = {m["id"]:m for m in protocol["models"]["twenty_hf_models"]}
    units = []
    for model_id in spec["models"]:
        model = models[model_id]
        for method in METHOD_NAMES:
            for library in spec["library_conditions"]:
                cells = [dict(track="S", stage="core", model_id=model_id, model=model["model"],
                    model_revision=model["revision"], method=method, library=library,
                    task_id=t.id, task_sha256=t.sha256, task_algorithm=t.algorithm, seed=seed)
                    for t in tasks.heldout for seed in spec["evaluation_seeds"]]
                units.append(dict(id=f"S-{model_id}-{method}-{int(library)}", model_id=model_id,
                    method=method, library=library, learned=method in LEARNERS,
                    fit_seed=spec["fit_seed"], cells=cells))
    return units


def phase_costs(path):
    """Count only bounded provider calls, separating fit from each test episode."""
    active, stack = None, []
    costs = defaultdict(lambda: dict(calls=0, generated_tokens=0, prompt_tokens=0))
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event["kind"]
        if kind in ("fit_started", "episode_started"):
            if stack:
                raise ValueError("phase changed during an unfinished provider call")
            active = (("fit", event["method"], event["library"]) if kind == "fit_started" else
                      ("test", event["method"], event["library"], event["task_id"], event["seed"]))
        elif kind == "call_started":
            if active is None:
                raise ValueError("provider call has no fit/test identity")
            mode = event["backend_mode"]
            if mode not in ("nested", "bounded"):
                raise ValueError("formal accounting requires a bounded provider")
            stack.append(dict(mode=mode, response=None))
            if mode == "bounded":
                costs[active]["calls"] += 1
        elif kind == "response_received":
            if not stack:
                raise ValueError("response without a started call")
            if stack[-1]["mode"] == "bounded":
                usage = event["response"].get("usage", {})
                if any(type(usage.get(k)) is not int or usage[k] < 0
                       for k in ("completion_tokens", "prompt_tokens")):
                    raise ValueError("exact nonnegative provider usage required")
                if stack[-1]["response"] is not None:
                    raise ValueError("duplicate provider response")
                stack[-1]["response"] = usage
        elif kind in ("call_completed", "call_failed"):
            if not stack or stack[-1]["mode"] != event["backend_mode"]:
                raise ValueError("call event nesting differs")
            call = stack.pop()
            if call["mode"] == "bounded":
                usage = call["response"]
                tokens = event.get("tokens") if kind == "call_completed" else event.get("token_cost")
                if usage is None or tokens != usage["completion_tokens"]:
                    raise ValueError("incomplete provider cost cannot be scored as zero")
                costs[active]["generated_tokens"] += tokens
                costs[active]["prompt_tokens"] += usage["prompt_tokens"]
    if stack:
        raise ValueError("unfinished provider call")
    return dict(costs)


def validate_unit(report, fits, unit, costs):
    """Verify every cell, one frozen fit, state reuse, and exact provider costs."""
    from fin_skills.rsi.run_record import digest
    report._validate()
    keys = ("method", "library", "task_id", "seed")
    expected = {tuple(c[k] for k in keys):c for c in unit["cells"]}
    actual = {tuple(r[k] for k in keys):r for r in report.rows}
    if len(actual) != len(report.rows) or set(actual) != set(expected):
        raise ValueError("seven-method unit must contain every frozen task and seed")
    if len(fits) != 1:
        raise ValueError("each method/model/library unit requires exactly one fit record")
    fit = fits[0]
    if (fit["method"] != unit["method"] or fit["library"] is not unit["library"]
            or fit["model_id"] != unit["cells"][0]["model"]
            or fit["state_sha256"] != digest(fit["state"])):
        raise ValueError("fit identity or frozen state differs")
    zero = dict(calls=0, generated_tokens=0, prompt_tokens=0)
    fit_key = ("fit", unit["method"], unit["library"])
    fit_cost = costs.get(fit_key, zero)
    if any(fit["usage"].get(k, 0) != fit_cost[k] for k in ("calls", "generated_tokens")):
        raise ValueError("fit costs differ from provider ledger")
    if not unit["learned"] and any(fit_cost.values()):
        raise ValueError("nonlearning method has unexpected fit costs")
    used = {fit_key}
    for key, row in actual.items():
        cell = expected[key]
        if (row["model_id"] != cell["model"] or row["task_sha256"] != cell["task_sha256"]
                or row["state_sha256"] != fit["state_sha256"]):
            raise ValueError("test model, task or frozen fitted state changed")
        cost_key = ("test", *key)
        used.add(cost_key)
        cost = costs.get(cost_key, zero)
        if any(row["usage"][k] != cost[k] for k in ("calls", "generated_tokens")):
            raise ValueError("test costs differ from provider ledger")
    if set(costs)-used:
        raise ValueError("unplanned provider phase")
    return dict(unit_id=unit["id"], episodes=len(actual), fit_cost=dict(fit_cost),
        test_cost={k:sum(costs.get(("test", *key), zero)[k] for key in actual) for k in zero},
        state_sha256=fit["state_sha256"],
        amortized_fit_cost_per_test_episode={k:v/len(actual) for k,v in fit_cost.items()},
        fit_usage=fit["usage"])


def complete_summary(rows, units, unit_receipts):
    """All five models and all seven methods are required before quality summaries."""
    keys = ("model_id", "method", "library", "task_id", "seed")
    cells = [dict(c, model_id=c["model"]) for u in units for c in u["cells"]]
    expected = {tuple(c[k] for k in keys):c for c in cells}
    actual = {tuple(r[k] for k in keys):r for r in rows}
    if len(expected) != len(cells) or len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("complete seven-method population required; missing cells are not zero")
    if (len(unit_receipts) != len(units)
            or {r["unit_id"] for r in unit_receipts} != {u["id"] for u in units}):
        raise ValueError("complete one-fit-per-unit cost records required")
    if any(type(r.get("valid")) is not bool or r["task_sha256"] != expected[k]["task_sha256"]
           or r["task_algorithm"] != expected[k]["task_algorithm"] for k,r in actual.items()):
        raise ValueError("invalid grade or task identity")
    groups = []
    for method in METHOD_NAMES:
        for library in (False, True):
            members = [r for r in rows if r["method"] == method and r["library"] == library]
            groups.append(dict(method=method, library=library, episodes=len(members),
                valid=sum(r["valid"] for r in members),
                statuses=dict(Counter(r["status"] for r in members)),
                by_model={model:dict(episodes=sum(r["model_id"] == model for r in members),
                    valid=sum(r["valid"] for r in members if r["model_id"] == model),
                    statuses=dict(Counter(r["status"] for r in members if r["model_id"] == model)))
                    for model in sorted({r["model_id"] for r in members})},
                by_algorithm={alg:sum(r["valid"] for r in members if r["task_algorithm"] == alg)/
                    sum(r["task_algorithm"] == alg for r in members)
                    for alg in sorted({r["task_algorithm"] for r in members})}))

    def contrast(method, library_effect):
        effects = defaultdict(list)
        for key, row in actual.items():
            if row["method"] != method or (library_effect and not row["library"]):
                continue
            model, _, library, task, seed = key
            comparator = (model, method if library_effect else "no_improve",
                          False if library_effect else library, task, seed)
            effects[row["task_algorithm"]].append(int(row["valid"])-int(actual[comparator]["valid"]))
        means = {a:sum(v)/len(v) for a,v in sorted(effects.items())}
        return dict(method=method, comparison="library-minus-scratch" if library_effect else
                    "method-minus-no_improve; equal average over both library conditions",
                    algorithm_effects=means, effect=sum(means.values())/len(means),
                    algorithm_blocks=len(means), descriptive_only=True)

    return dict(episodes=len(rows), units=len(units), learned_fits=sum(u["learned"] for u in units),
        groups=groups, contrasts=[contrast(m, False) for m in METHOD_NAMES if m != "no_improve"]+
            [contrast(m, True) for m in METHOD_NAMES], unit_costs=unit_receipts,
        interpretation="fixed public algorithms; models, variants and seeds are not independent task units",
        independent_population_inference=False, publication_ready=False)
