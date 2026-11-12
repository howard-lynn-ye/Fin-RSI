"""Backend wrapper for P-adopt: a context overflow ends one episode, not the whole run.

TransformersChat refuses a request whose prompt plus output reservation exceeds the model's
native context, before generating anything. In the frozen study a dedicated guard recorded
that case; here the refusal becomes BudgetExceeded("context"), which the methods already
record as the episode status ``budget:context``. Message digests of every forwarded request
are kept so a run can show which first prompts reached the model.
"""
from fin_skills.rsi.meter import BudgetExceeded
from fin_skills.rsi.run_record import digest

OVERFLOW = "context budget exceeded; no silent truncation"


class ContextAsBudget:
    def __init__(self, backend):
        self.backend = backend
        self.forwarded = []           # digests of complete message lists sent to the model
        self.overflows = 0

    def __call__(self, messages):
        raise RuntimeError("bounded generation required")

    def describe(self):
        return self.backend.describe()

    def generate_bounded(self, messages, *, max_new_tokens, seed):
        try:
            response = self.backend.generate_bounded(messages, max_new_tokens=max_new_tokens,
                                                     seed=seed)
        except ValueError as exc:
            if str(exc) != OVERFLOW:
                raise
            self.overflows += 1
            raise BudgetExceeded("context", max_new_tokens, 0) from exc
        self.forwarded.append(digest(messages))
        return response
