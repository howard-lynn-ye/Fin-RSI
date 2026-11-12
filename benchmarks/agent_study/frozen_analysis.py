"""Prospective source-balanced summaries; seeds and models never increase document N."""
from collections import Counter, defaultdict
import itertools

import numpy as np

UNIT_TAGS = ("question-explicit-percent-or-bp", "question-explicit-currency-or-scale",
             "question-explicit-ratio", "question-explicit-count", "context-only-unit",
             "multiple-or-ambiguous", "unspecified")


def match_complete(rows, cells):
    def key(row):
        return row["model_id"], row["method"], row["library"], row["task_id"], row["seed"]
    planned = {key(dict(cell, model_id=cell["model"])):cell for cell in cells}
    actual = {key(row):row for row in rows}
    if (not planned or len(planned) != len(cells) or len(actual) != len(rows)
            or set(planned) != set(actual)):
        raise ValueError("complete fixed population required; missing cells are not zero scores")
    if any(type(r.get("valid")) is not bool or r["task_sha256"] != planned[k]["task_sha256"]
           for k, r in actual.items()):
        raise ValueError("invalid grade or changed task identity")
    return actual


def source_mean(values, metadata, multiplicities=None):
    """Two prespecified source means, each with weight 1/2; None for absent source."""
    totals, counts = defaultdict(float), defaultdict(float)
    for task_id, value in values.items():
        item = metadata[task_id]
        weight = 1 if multiplicities is None else multiplicities.get(task_id, 0)
        totals[item["source_task"]] += value * weight
        counts[item["source_task"]] += weight
    if any(counts[source] == 0 for source in ("CodeFinQA", "CodeTAT-QA")):
        return None
    return .5 * sum(totals[source] / counts[source] for source in ("CodeFinQA", "CodeTAT-QA"))


def tolerance_sensitivity(rows, cells, selected):
    """Descriptive alternative grade, with the complete original pairing and weights."""
    match_complete(rows, cells)
    metadata = {item["task_id"]:item for item in selected}
    if set(metadata) != {row["task_id"] for row in rows}:
        raise ValueError("enrollment metadata does not cover every result")
    if any(type(row.get("relative_tolerance_1pct")) is not bool for row in rows):
        raise ValueError("every original final output needs the prespecified sensitivity grade")
    if any(row["valid"] and not row["relative_tolerance_1pct"] for row in rows):
        raise ValueError("exact pass must imply sensitivity pass")

    def summarize(members):
        rates, transitions = {}, {}
        for lib in (False, True):
            by_task = defaultdict(list)
            for row in members:
                if row["library"] == lib:
                    by_task[row["task_id"]].append(row)
            rates[str(lib)] = {metric:source_mean({task:sum(r[field] for r in group)/len(group)
                for task, group in by_task.items()}, metadata)
                for metric, field in (("exact", "valid"), ("relative_1pct", "relative_tolerance_1pct"))}
            group = [row for row in members if row["library"] == lib]
            transitions[str(lib)] = dict(episodes=len(group),
                exact_pass=sum(row["valid"] for row in group),
                exact_fail_relative_pass=sum(not row["valid"] and row["relative_tolerance_1pct"] for row in group),
                both_fail=sum(not row["relative_tolerance_1pct"] for row in group))
        effects = {metric:rates["True"][metric]-rates["False"][metric] for metric in ("exact", "relative_1pct")}
        direction = lambda x: (x > 0)-(x < 0)
        return dict(source_balanced_rates=rates, source_balanced_effects=effects,
                    effect_direction_changed=direction(effects["exact"]) != direction(effects["relative_1pct"]),
                    transitions=transitions)

    return dict(name="prespecified relative-tolerance sensitivity analysis", **summarize(rows),
        by_model={model:summarize([r for r in rows if r["model_id"] == model])
                  for model in sorted({r["model_id"] for r in rows})},
        dependency_blocks=len({item["dependency_block"] for item in selected}),
        independent_population_inference=False, confirmatory_tests=False,
        interpretation="Numeric proximity, not proof of semantic correctness or a specific error cause; primary exact grades unchanged")


def blocked_interval(values, metadata, field, *, draws=10000, seed=760011):
    blocks = defaultdict(list)
    for task_id in values:
        blocks[metadata[task_id][field]].append(task_id)
    names = sorted(blocks)
    if len(names) < 2:
        return dict(interval=None, blocks=len(names), defined_draws=0, undefined_draws=draws,
                    reason="fewer than two dependency blocks")
    # One common draw across both sources; resampling each source separately would
    # break shared issuer dependence. Missing-source draws remain explicitly counted.
    sources = ("CodeFinQA", "CodeTAT-QA")
    sums = np.zeros((len(names), 2))
    counts = np.zeros((len(names), 2))
    for i, name in enumerate(names):
        for task_id in blocks[name]:
            j = sources.index(metadata[task_id]["source_task"])
            sums[i, j] += values[task_id]
            counts[i, j] += 1
    rng = np.random.default_rng(seed)
    distribution, undefined = [], 0
    for _ in range(draws):
        chosen = rng.integers(0, len(names), size=len(names))
        denominator = counts[chosen].sum(axis=0)
        if np.any(denominator == 0):
            undefined += 1
        else:
            distribution.append(float(np.mean(sums[chosen].sum(axis=0) / denominator)))
    interval = np.quantile(distribution, [.025, .975]).tolist() if distribution else None
    return dict(interval=interval, blocks=len(names), defined_draws=len(distribution),
                undefined_draws=undefined, draws=draws, seed=seed)


def external(rows, cells, selected, *, draws=10000):
    match_complete(rows, cells)
    metadata = {item["task_id"]:item for item in selected}
    if set(metadata) != {r["task_id"] for r in rows}:
        raise ValueError("enrollment metadata does not cover every result")
    condition = {lib:defaultdict(list) for lib in (False, True)}
    models = defaultdict(list)
    for row in rows:
        condition[row["library"]][row["task_id"]].append(row)
        models[row["model_id"]].append(row)
    means = {lib:{task:sum(r["valid"] for r in members)/len(members)
                  for task, members in by_task.items()} for lib, by_task in condition.items()}
    effects = {task:means[True][task]-means[False][task] for task in metadata}
    population = Counter(item["source_task"] for item in selected)
    weights = defaultdict(float)
    for item in selected:
        weights[item["dependency_block"]] += .5 / population[item["source_task"]]
    blocked = blocked_interval(effects, metadata, "dependency_block", draws=draws)
    template = blocked_interval(effects, metadata, "template_sha256", draws=draws)
    remaining = []
    for template_id in {item["template_sha256"] for item in selected}:
        value = source_mean({k:v for k,v in effects.items()
                             if metadata[k]["template_sha256"] != template_id}, metadata)
        if value is not None:
            remaining.append(value)
    strata = {}
    for tag in UNIT_TAGS:
        ids = [key for key, item in metadata.items() if tag in item["units"]["tags"]]
        strata[tag] = dict(questions=len(ids), source_counts=dict(Counter(metadata[k]["source_task"] for k in ids)),
            source_balanced_effect=source_mean({k:effects[k] for k in ids}, metadata),
            question_mean_effect=sum(effects[k] for k in ids)/len(ids) if ids else None,
            condition_rates={str(lib):source_mean({k:means[lib][k] for k in ids}, metadata)
                             for lib in (False, True)},
            error_patterns=dict(Counter(r.get("numeric_error_pattern", "not_derived")
                for r in rows if r["task_id"] in ids)))
    by_model = {}
    for model, members in sorted(models.items()):
        pairs = defaultdict(dict)
        for row in members:
            pairs[(row["task_id"], row["seed"])][row["library"]] = int(row["valid"])
        differences = defaultdict(list)
        for (task_id, _), pair in pairs.items():
            if set(pair) != {False, True}:
                raise ValueError("external paired condition missing")
            differences[task_id].append(pair[True]-pair[False])
        model_rates = {}
        for lib in (False, True):
            by_task = defaultdict(list)
            for row in members:
                if row["library"] == lib:
                    by_task[row["task_id"]].append(row["valid"])
            model_rates[str(lib)] = source_mean({k:sum(v)/len(v) for k,v in by_task.items()}, metadata)
        by_model[model] = dict(effect=source_mean({k:sum(v)/len(v) for k,v in differences.items()}, metadata),
                               rates=model_rates, episodes=len(members))
    costs = {str(lib):dict(episodes=sum(len(v) for v in condition[lib].values()),
        calls=sum(r["usage"]["calls"] for group in condition[lib].values() for r in group),
        generated_tokens=sum(r["usage"]["generated_tokens"] for group in condition[lib].values() for r in group),
        prompt_tokens=sum(r.get("derived_prompt_tokens", 0) for group in condition[lib].values() for r in group),
        prompt_tokens_complete=all("derived_prompt_tokens" in r for group in condition[lib].values() for r in group),
        accepted_wrong=sum(r["accepted"] and not r["valid"] for group in condition[lib].values() for r in group))
        for lib in (False, True)}
    exposure = {}
    for field in ("same_context", "same_template", "same_verified_issuer"):
        ids = [key for key, item in metadata.items() if not item["development_exposure"][field]]
        exposure["exclude_"+field] = dict(questions=len(ids),
            source_counts=dict(Counter(metadata[key]["source_task"] for key in ids)),
            source_balanced_effect=source_mean({key:effects[key] for key in ids}, metadata))
    return dict(questions=len(metadata), episodes=len(rows), source_counts=dict(population),
        source_balanced_rates={str(lib):source_mean(values, metadata) for lib, values in means.items()},
        source_balanced_effect=source_mean(effects, metadata), dependency_bootstrap=blocked,
        effective_groups=1/sum(w*w for w in weights.values()), unresolved_weight=weights.get("unresolved", 0),
        template_sensitivity=template, leave_template_out_range=[min(remaining), max(remaining)] if remaining else None,
        by_source={source:sum(effects[k] for k in metadata if metadata[k]["source_task"] == source)/n
                   for source, n in population.items()},
        unit_strata=strata, model_effects=by_model, costs=costs, development_exposure_sensitivity=exposure,
        inference="descriptive fixed public benchmark; bootstrap assumes its declared blocks",
        independent_population_inference=False, publication_ready=False)


def positions(rows, cells, contrasts):
    match_complete(rows, cells)
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["method"], row["library"])][(row["model_id"], row["task_id"], row["seed"])] = row
    result = []
    for left, right in contrasts:
        a, b = groups[tuple(left)], groups[tuple(right)]
        if set(a) != set(b):
            raise ValueError("position contrast has unmatched cells")
        blocks = defaultdict(list)
        for key, row in a.items():
            blocks[row["task_algorithm"]].append(int(row["valid"])-int(b[key]["valid"]))
        effects = {key:sum(values)/len(values) for key, values in sorted(blocks.items())}
        mean = sum(effects.values())/len(effects)
        if len(effects) > 12:
            raise ValueError("this exact descriptive analysis is for the frozen small algorithm suite")
        flips = [abs(sum(value*sign for value,sign in zip(effects.values(), signs))/len(effects))
                 for signs in itertools.product((-1, 1), repeat=len(effects))]
        result.append(dict(left=left, right=right, effect=mean, algorithm_effects=effects,
            algorithm_blocks=len(effects), p_sign_flip=sum(v >= abs(mean)-1e-12 for v in flips)/len(flips),
            p_min=2/len(flips), interpretation="descriptive public algorithms; fixed model/seed means"))
    previous = 0
    for i, row in enumerate(sorted(result, key=lambda r:r["p_sign_flip"])):
        previous = min(1, max(previous, (len(result)-i)*row["p_sign_flip"]))
        row["p_holm"] = previous
    summaries = []
    for (method, library), group in sorted(groups.items()):
        members = list(group.values())
        summaries.append(dict(method=method, library=library, episodes=len(members),
            valid=sum(r["valid"] for r in members), first_valid=sum(r["first_valid"] for r in members),
            accepted_wrong=sum(r["accepted"] and not r["valid"] for r in members),
            calls=sum(r["usage"]["calls"] for r in members),
            generated_tokens=sum(r["usage"]["generated_tokens"] for r in members),
            statuses=dict(Counter(r["status"] for r in members))))
    return dict(episodes=len(rows), contrasts=result, groups=summaries, independent_population_inference=False,
                publication_ready=False)
