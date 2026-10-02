# FinQA interface correction: prospective protocol v2

The September 23--24 runs and their original scores remain frozen. This amendment
applies to new runs only and supersedes the final-answer presentation rule in the old
RAG and reranking protocols. It is not a new result or a completed model qualification.

The ReAct prompt now gives an explicit final-response example, prohibits Markdown and
trailing prose, and instructs the model to finish after a successful calculation rather
than repeat it. Numbers and labels in the example are explicitly unrelated to the task.
Model-call and token budgets, numerical grading, and the required two string fields
(`program`, `citations`) remain unchanged.

New inference protocols declare `answer_format: json-or-single-fence-v2`. Scoring accepts
an exact JSON object or one complete JSON/unlabeled Markdown fence around that object.
It does not search arbitrary prose, select one of multiple answers, repair programs,
fill missing fields, accept duplicate keys, or convert a tool receipt into a final answer.
The raw answer remains saved. Strict JSON compliance is reported separately from numerical
correctness. Protocols without this field retain `strict-json-v1`; the audit supports
archived v1 source without changing its interface or grades.

Score rows also retain the original episode error, a missing-final flag, and the number
of length-limited responses. A null final answer remains a miss but is no longer diagnosed
only through the secondary JSON TypeError. Retrieval, execution and output compliance
must be analyzed separately; a well-formatted answer is not evidence of correct retrieval.

Before another production comparison, use a fresh qualification root with matching source
hashes and verify actual model finalization on development questions, including a simple
known-answer control. Scripted software fixtures alone do not qualify model behavior.
Record tokenizer configuration and resolve the prior Mistral regex warning on development
inputs. Run all compared answer arms with the same prompt, parser and runtime versions;
do not compare a new reranking arm to the old RAG control as a matched experiment.

The inspected public test questions are development-exposed. Any later use must be labeled
as regression evidence, with an independently frozen set needed for a new final claim.
