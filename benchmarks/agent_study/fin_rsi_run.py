"""Single parameterized Fin-RSI runner (replaces per-round v1/v2/v3 launch scripts).

Runs sealed initial messages from a row-binding file through one agent loop with a chosen
interface and backend, and writes one directory per row plus summary.json.

    python -m benchmarks.agent_study.fin_rsi_run \
        --rows results/.../inputs/row-bindings-v1.json --condition C_unchanged \
        --model general14b=Qwen/Qwen2.5-14B-Instruct@e5147e7a8cef30e2ffa759518dc9f6c45e7940ca \
        --interface tolerant --out results/fin_rsi/runs/<name>

Backends: ``hf`` (TransformersChat on the allocated GPU) or ``replay:<attempt_root>`` (replays
saved responses of the same unit/model from an earlier run; CPU smoke and turn-1
counterfactuals; later turns are not counterfactual because feedback differs).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmarks.agent_study import fin_rsi_interface as interface  # noqa: E402
from benchmarks.agent_study import trading_capabilities as caps  # noqa: E402
from benchmarks.agent_study import trading_runtime_v6 as runtime  # noqa: E402

CALL_NOTICE = '\nCurrent call 1 of 8; 8 calls remain including this one. Finish with submit(weights).'
TRUNCATED_MESSAGE = ('Your response reached the {cap} output-token limit and was NOT executed. '
                     'Use one shorter JSON tool call or one concise complete Python fence; '
                     'omit explanations.')
FORMAT_MESSAGE = ('Use exactly one tool JSON object or ONE complete Python fence. '
                  'Multiple code blocks are not executed; combine code into one fence. '
                  'Return the call, not a multi-step prose plan.')
# System-prompt variants appended before the per-turn call notice; 'sealed' leaves it unchanged.
# 'algo-menu' (2026-10-18 skill-use diagnosis): over 48 saved rows of jobs 2367319/443/577 the
# models wrote run_algorithm in 30 rows, never after list_algorithms, and 253/255 algorithm_id
# values were invented (198x mean_variance_optimizer); the sealed prompt names no algorithm ID.
# When list_algorithms did return min_variance, the next turn used it in 2 of 3 cases.
SYSTEM_VARIANTS = {
    'sealed': '',
    'algo-menu': '''

## Portfolio algorithms ready in this sandbox (exact algorithm_id values)
- equal_weight: equal weight
- inverse_volatility: inverse volatility
- min_variance: long-only minimum variance
- hrp: hierarchical risk parity
- cross_sectional_momentum: positive-momentum top-k; unallocated balance is cash
Only these IDs return portfolio weights; other names fail. describe_algorithm(ID) shows
defaults; list_algorithms(task="portfolio") pages the full registry.
Complete example in ONE Python fence (replace the ID with your choice from the list):
```python
reply = run_algorithm(algorithm_id="hrp", lookback=252)
assert reply["ok"], reply
weights = reply["result"]["weights"]
print(weights)
submit(weights)
```
You may inspect, modify or reject the proposed weights before submit.''',
}
# 'algo-menu-read' (2026-10-19): algo-menu plus one neutral discovery sentence that names neither
# a skill nor a section; 2367799 (algo-menu) chose min_variance 11/11 and read no skill.
SYSTEM_VARIANTS['algo-menu-read'] = SYSTEM_VARIANTS['algo-menu'] + (
    '\nBefore choosing a method, find and read the installed skill guidance on choosing among '
    'these portfolio methods (list_skills / read_skill; follow next_offset until the page list '
    'is complete).')
# 'algo-menu-read-first' (2026-10-19, option A): 2367810 ignored that trailing sentence 0/12, and
# 5/12 rows submitted in turn 1 through the one-fence example. Same sentence and menu; the sentence
# moves before the menu and the example becomes two turns (discover, then run and submit).
_MENU, _EXAMPLE = SYSTEM_VARIANTS['algo-menu'].split(
    'Complete example in ONE Python fence (replace the ID with your choice from the list):\n')
_READ_SENTENCE = (
    'Before choosing a method, find and read the installed skill guidance on choosing among '
    'these portfolio methods (list_skills / read_skill; follow next_offset until the page list '
    'is complete).')
_TWO_STEP = (
    _MENU + 'Two-step example, one call per turn.\n'
    'Turn 1, discover guidance (JSON): {"tool":"list_skills","arguments":{"query":"optimizer"}}\n'
    'Later turn, after reading, ONE Python fence (replace the ID with your choice from the list):\n'
    + _EXAMPLE)
SYSTEM_VARIANTS['algo-menu-read-first'] = '\n\n' + _READ_SENTENCE + _TWO_STEP
# 'algo-menu-read-user' (2026-10-19, option D): 2367839 ignored the read sentence 0/12 in the system
# prompt, and across 48 algo-menu rows describe_algorithm (also a system-prompt "before X" rule) was
# called 0 times. Same system text as algo-menu-read-first minus the sentence; the identical sentence
# is appended to the end of the initial user (task) message instead.
SYSTEM_VARIANTS['algo-menu-read-user'] = _TWO_STEP
USER_SUFFIXES = {'algo-menu-read-user': '\n' + _READ_SENTENCE}
# 'algo-menu-two-step' (2026-10-19, conditional-task design): algo-menu-read-user without the user
# sentence, i.e. the same system text; the "no read instruction" arm differs from D by one sentence.
SYSTEM_VARIANTS['algo-menu-two-step'] = _TWO_STEP
# I18 (2026-10-19): decision channel. 'free' = control (submit(weights)), 'structured' = the decision
# must be allocate(method, max_weight, max_turnover) (fin_rsi_interface.DecisionChannelMixin). Each
# channel appends its own closing user paragraph (118 / 121 Qwen tokens); 'structured' also swaps
# the example's last line and the per-turn notice. Without --decision-channel nothing changes.
CHANNEL_PARAGRAPHS = {
    'free': 'Decision channel: finish with submit(weights), one ticker:weight dictionary for the instruments listed above, e.g. submit({"SPY": 0.5, "IEF": 0.5}) for shape only. The weights may come from one of the five IDs above, from your own calculation on the visible history, or from changing either of them by hand; each value is a fraction of the portfolio (0.25 means 25%), they must be long-only, and they are normalized to sum to 1. It then submits those weights. allocate(...) is not available.',
    'structured': 'Decision channel: finish with allocate(method, max_weight=1.0, max_turnover=1.0), method being one of the five IDs above. It runs that method on the visible history, caps every weight at max_weight (the excess goes to the other instruments), and caps one-way turnover from the current holdings at max_turnover by moving only part of the way. It then submits the result. max_weight: 0.10 to 1.0; max_turnover: 0.0 to 1.0. submit(weights) is not available.',
}
_EXAMPLE_TAIL = ('weights = reply["result"]["weights"]\nprint(weights)\nsubmit(weights)\n```\n'
                 'You may inspect, modify or reject the proposed weights before submit.')
assert _EXAMPLE.endswith(_EXAMPLE_TAIL)
STRUCTURED_SYSTEM_VARIANTS = {'algo-menu-two-step': _TWO_STEP[:-len(_EXAMPLE_TAIL)] + (
    'print(reply["result"]["weights"])\nallocate(method="hrp")\n```\n'
    'You may inspect the proposed weights before allocate.')}
CHANNEL_FINISH = {'': 'submit(weights)', 'free': 'submit(weights)',
                  'structured': 'allocate(method, max_weight, max_turnover)'}
# I15 option (c), 2026-10-19: is the inline evidence used under the current best config (S0 =
# algo-menu-read-user)? Same system text and D sentence; one more sentence appended to the user
# message. The pointer allows "none relevant"; the control has the same position, length (148 chars)
# and word count (26), and points at market data instead of evidence records.
_EVIDENCE_POINTER = ("The dated evidence records above may or may not bear on the weights; before "
                     "submitting, name the record ids you relied on, or write 'none relevant'.")
_EVIDENCE_NEUTRAL = ("The dated market quote series above may or may not bear on the weights; before "
                     "submitting, name the tickers you relied on, or write 'none relevant'.")
assert len(_EVIDENCE_POINTER) == len(_EVIDENCE_NEUTRAL)
SYSTEM_VARIANTS['algo-menu-read-user-evidence'] = _TWO_STEP
USER_SUFFIXES['algo-menu-read-user-evidence'] = '\n' + _READ_SENTENCE + '\n' + _EVIDENCE_POINTER
SYSTEM_VARIANTS['algo-menu-read-user-neutral'] = _TWO_STEP
USER_SUFFIXES['algo-menu-read-user-neutral'] = '\n' + _READ_SENTENCE + '\n' + _EVIDENCE_NEUTRAL

# Task variants (docs/fin_rsi_sessions/claude_20261019_conditional_task_design.md): replace the one
# objective sentence of the sealed user message; 'sealed' leaves the message untouched. No method names.
TASK_OBJECTIVE = ('Objective: maximize net risk-adjusted return over the holding period; there is no '
                  'benchmark to track and cash earns zero.')
_INVESTED = ' Stay fully invested: weights must sum to 1 (no cash).'
TASK_VARIANTS = {
    'sealed': None,
    'min-vol': ("Objective: minimize the volatility of the portfolio's daily returns over the holding "
                'period; return is secondary.' + _INVESTED),
    'equal-risk': ('Objective: every listed instrument should be held and should contribute an equal '
                   "share of the portfolio's total risk (variance) over the holding period." + _INVESTED),
    'cap30': ('Objective: maximize net risk-adjusted return over the holding period, subject to a hard '
              'limit: no single instrument may exceed 30% of the portfolio.' + _INVESTED),
    'turnover10': ('Objective: maximize net risk-adjusted return over the holding period, subject to a '
                   'hard limit: one-way turnover from the current holdings, 0.5 \u00d7 sum of absolute '
                   'weight changes, must not exceed 10%.' + _INVESTED),
}


def apply_task_variant(messages, name):
    if TASK_VARIANTS[name] is None:
        return messages
    user = messages[1]['content']
    if user.count(TASK_OBJECTIVE) != 1:
        raise RuntimeError(f'task variant {name}: objective sentence found {user.count(TASK_OBJECTIVE)} times')
    return [messages[0], dict(messages[1], content=user.replace(TASK_OBJECTIVE, TASK_VARIANTS[name]))] + messages[2:]

# Generation overrides (I17): Qwen3 model card settings for thinking mode; 'default' changes nothing.
SAMPLING = {'default': None, 'qwen3-thinking': dict(temperature=0.6, top_p=0.95, top_k=20)}


class Controller(caps.Controller):
    worker_module = 'benchmarks.agent_study.fin_rsi_interface'


ARM_FLAGS = ('algorithm_hint', 'persistent_python', 'python_only_hint', 'skill_name_hint')


class ReplayBackend:
    """Saved responses for one source row, in turn order."""

    def __init__(self, attempt_root, source_condition):
        self.root, self.source_condition, self.queue = Path(attempt_root), source_condition, []

    def start_row(self, row):
        source = f"{row['unit_id']}__{row['model_tag']}__{self.source_condition}"
        directory = self.root / 'rows' / source / 'provider-responses'
        self.queue = [json.loads(p.read_text())['response']
                      for p in sorted(directory.glob('turn-*.json'))]
        self.source_row = source

    def __call__(self, messages):
        if not self.queue:
            raise ValueError('replay exhausted')
        return self.queue.pop(0)


class HFBackend:
    def __init__(self, model_id, revision, cap, sampling=None):
        from benchmarks.agent_study.transformers_chat import TransformersChat
        self.chat = TransformersChat(model_id, revision, max_tokens=cap, seed=0, sampling=sampling)

    def start_row(self, row):
        self.chat.seed, self.chat.calls = row['sampling_seed'], 0

    def __call__(self, messages):
        return self.chat(messages)


def decide(backend, controller, messages, iface, max_turns, cap, system_variant='sealed',
           strip_thinking=False, decision_channel=''):
    """trading_runtime_v6.decide over sealed initial messages; parser chosen by ``iface``."""
    system = messages[0]['content']
    if not system.endswith(CALL_NOTICE):
        raise RuntimeError('sealed system message lacks the turn-1 call notice')
    base = system[:-len(CALL_NOTICE)] + (STRUCTURED_SYSTEM_VARIANTS[system_variant] if
                                         decision_channel == 'structured' else SYSTEM_VARIANTS[system_variant])
    history = [dict(messages[0]), dict(messages[1])]
    history[1]['content'] += USER_SUFFIXES.get(system_variant, '')
    if decision_channel:
        history[1]['content'] += '\n' + CHANNEL_PARAGRAPHS[decision_channel]
    finish = CHANNEL_FINISH[decision_channel]
    turns, target = [], None
    for index in range(max_turns):
        remaining = max_turns - index
        history[0]['content'] = (base + f'\nCurrent call {index + 1} of {max_turns}; '
                                 f'{remaining} calls remain including this one. Finish with {finish}.')
        if remaining == 1:
            history[0]['content'] += ' This is the last call; no later response can execute.'
        try:
            response = backend(history)
        except ValueError as exc:
            turns.append(dict(turn=index + 1, parse='backend-stop', error=str(exc)))
            break
        choice = response['choices'][0]
        text, truncated = choice['message']['content'], choice['finish_reason'] == 'length'
        call, status, candidates = interface.extract_call(iface, text, truncated)
        if call is None:
            result = dict(ok=False, error=TRUNCATED_MESSAGE.format(cap=cap)
                          if status == 'generation-truncated' or (strip_thinking and truncated and
                                                                   status == 'reasoning-incomplete')
                          else FORMAT_MESSAGE)
        elif call.get('tool') == 'submit' and decision_channel == 'structured':
            result = dict(ok=False, error=interface.SUBMIT_REDIRECT)
        elif call.get('tool') == 'submit':
            args = call.get('arguments')
            weights = args.get('weights', args) if isinstance(args, dict) else None
            target, problem = runtime.validate_weights(weights)
            result = dict(ok=target is not None, submission=target, error=problem)
        else:
            result = controller.call(call.get('tool'), call.get('arguments'))
            if result.get('ok') and result.get('submission') is not None:
                target, problem = runtime.validate_weights(result['submission'])
                if problem:
                    result = dict(ok=False, error=problem)
        status_fields = runtime.execution_status(result)
        feedback = status_fields['model_feedback']
        note = interface.tolerant_note(status, candidates) if iface == 'tolerant' else None
        if note:
            feedback = note + '\n' + feedback
        usage = response.get('usage') or {}
        turns.append(dict(turn=index + 1, parse=status, candidates=candidates,
                          tool=call.get('tool') if call else None, ok=bool(result.get('ok')),
                          error=(result.get('error') or '')[:300] or None,
                          finish_reason=choice['finish_reason'],
                          input_tokens=usage.get('prompt_tokens'),
                          output_tokens=usage.get('completion_tokens'),
                          generation_seconds=response.get('generation_seconds'),
                          response=text, feedback=feedback))
        if target is not None:
            break
        kept = text.split('</think>')[-1].lstrip('\n') if strip_thinking and '</think>' in text else text
        history += [dict(role='assistant', content=kept), dict(role='user', content=feedback)]
    return dict(submitted=target is not None, target=target, turns=turns,
                tool_calls=controller.calls)


def row_metrics(decision):
    turns, calls = decision['turns'], decision['tool_calls']
    ok = lambda name: sum(1 for c in calls if c.get('tool') == name and c.get('ok'))
    algo = [c for c in calls if c.get('tool') == 'run_algorithm' and c.get('ok')]
    return dict(
        submitted=decision['submitted'], turns=len(turns),
        executed_turns=sum(1 for t in turns if t.get('ok')),
        rejected_turns=sum(1 for t in turns if t.get('parse') not in (None, 'backend-stop')
                           and t.get('tool') is None),
        truncated_turns=sum(1 for t in turns if t.get('finish_reason') == 'length'),
        multi_action_turns=sum(1 for t in turns if (t.get('candidates') or 0) > 1),
        skill_reads_ok=ok('read_skill'), list_algorithms_ok=ok('list_algorithms'),
        describe_algorithm_ok=ok('describe_algorithm'), load_history_ok=ok('load_history'),
        run_algorithm_ok=len(algo), algorithm_ids=sorted({c.get('algorithm_id') for c in algo
                                                          if c.get('algorithm_id')}),
        algorithm_then_submit=bool(algo) and decision['submitted'],
        output_tokens=sum(t.get('output_tokens') or 0 for t in turns),
        generation_seconds=round(sum(t.get('generation_seconds') or 0 for t in turns), 3))


def parse_model(spec):
    tag, rest = spec.split('=', 1)
    model_id, _, revision = rest.partition('@')
    return dict(tag=tag, model_id=model_id, revision=revision or None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--rows', required=True, help='row-bindings JSON (rows + units)')
    p.add_argument('--condition', required=True, help='row condition whose sealed messages to use')
    p.add_argument('--model', action='append', required=True, help='tag=model_id@revision')
    p.add_argument('--interface', choices=interface.INTERFACES, required=True)
    p.add_argument('--backend', default='hf', help='hf | replay:<attempt_root>')
    p.add_argument('--replay-condition', default='C_unchanged')
    p.add_argument('--units', default='', help='comma-separated unit_ids (default: all)')
    p.add_argument('--seeds', default='', help='comma-separated sampling seeds (default: row seed)')
    p.add_argument('--system-variant', default='sealed',
                   help='comma-separated, run in sequence with one model load: '
                        + ', '.join(sorted(SYSTEM_VARIANTS)))
    p.add_argument('--task-variant', default='sealed',
                   help='comma-separated objective variants run in sequence: ' + ', '.join(TASK_VARIANTS))
    p.add_argument('--arm', choices=('raw', 'library'), default=None,
                   help="controller tool arm (default: the row's 'arm', else library)")
    p.add_argument('--algorithm-hint', action='store_true',
                   help='tolerant only: unknown algorithm_id errors list the valid ready ids')
    p.add_argument('--skill-search', choices=('substring', 'ranked'), default='substring',
                   help='list_skills: frozen substring match, or word-ranked with short rows')
    p.add_argument('--skill-overlay', default=None, metavar='SKILL=PATH',
                   help='tolerant only: read_skill(SKILL) serves the text in PATH (candidate patch)')
    p.add_argument('--snapshot-tool-log', action='store_true',
                   help='log a copy of each tool reply (agent code editing a reply in place no longer '
                        'rewrites tool_calls); not model-visible')
    p.add_argument('--skill-name-hint', action='store_true',
                   help='tolerant only: read_skill(<algorithm_id>) errors name the algorithm\'s skill')
    p.add_argument('--persistent-python', action='store_true',
                   help='tolerant only: run_python keeps picklable variables between calls')
    p.add_argument('--python-only-hint', action='store_true',
                   help='tolerant only: JSON calls to Python-only functions get a specific error')
    p.add_argument('--flag-arms', default='',
                   help='comma-separated interface-flag arms run in sequence with one model load, '
                        'e.g. persistent-python,python-only-hint ("+" joins flags inside an arm, '
                        '"none" = plain tolerant); added on top of the flags given directly')
    p.add_argument('--strip-history-thinking', action='store_true',
                   help='thinking models: keep only the text after the last </think> in the history (as Qwen3 '
                        'recommends); a turn cut off mid-thinking gets the truncation message')
    p.add_argument('--sampling', choices=sorted(SAMPLING), default='default',
                   help='generation overrides; default keeps temperature 0.1 / top_p 1.0 + model config')
    p.add_argument('--decision-channel', default='',
                   help='comma-separated decision channels run in sequence: free (submit(weights) + its '
                        'closing paragraph) and/or structured (allocate(method, max_weight, max_turnover) '
                        'only); default: unchanged prompt, no paragraph')
    p.add_argument('--max-turns', type=int, default=8)
    p.add_argument('--output-cap', type=int, default=1024)
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    variants = args.system_variant.split(',')
    task_variants = args.task_variant.split(',')
    channels = [c for c in args.decision_channel.split(',') if c] or ['']
    if set(channels) - {'', 'free', 'structured'} or len(set(channels)) < len(channels):
        p.error('unknown or repeated --decision-channel: ' + args.decision_channel)
    if 'structured' in channels and (args.interface != 'tolerant' or
                                     set(args.system_variant.split(',')) - set(STRUCTURED_SYSTEM_VARIANTS)):
        p.error('--decision-channel structured requires --interface tolerant and --system-variant '
                + ','.join(STRUCTURED_SYSTEM_VARIANTS))
    if set(task_variants) - set(TASK_VARIANTS) or len(set(task_variants)) < len(task_variants):
        p.error('unknown or repeated --task-variant: ' + args.task_variant)
    if set(variants) - set(SYSTEM_VARIANTS) or len(set(variants)) < len(variants):
        p.error('unknown or repeated --system-variant: ' + args.system_variant)
    for flag in ('algorithm_hint', 'persistent_python', 'python_only_hint', 'skill_name_hint',
                 'skill_overlay'):
        if getattr(args, flag) and args.interface != 'tolerant':
            p.error(f"--{flag.replace('_', '-')} requires --interface tolerant")

    binding = json.loads(Path(args.rows).read_text())
    units = {u['unit_id']: u for u in binding['units']}
    wanted = set(filter(None, args.units.split(',')))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['FIN_RSI_INTERFACE'] = args.interface
    os.environ['FIN_RSI_SKILL_SEARCH'] = args.skill_search
    overlay = None
    if args.skill_overlay:
        interface.load_skill_overlay(args.skill_overlay)  # fail fast on a bad skill or path
        overlay = dict(spec=args.skill_overlay, sha256=hashlib.sha256(
            Path(args.skill_overlay.partition('=')[2]).read_bytes()).hexdigest())
    os.environ['FIN_RSI_SKILL_OVERLAY'] = args.skill_overlay or ''
    os.environ['FIN_RSI_SNAPSHOT_TOOL_LOG'] = '1' if args.snapshot_tool_log else '0'
    flag_arms = []
    for name in filter(None, args.flag_arms.split(',')):
        chosen = [] if name == 'none' else [f.replace('-', '_') for f in name.split('+')]
        if set(chosen) - set(ARM_FLAGS):
            p.error(f'unknown --flag-arms flag in {name!r}; choose from ' + ', '.join(ARM_FLAGS))
        if chosen and args.interface != 'tolerant':
            p.error('--flag-arms flags require --interface tolerant')
        flag_arms.append((name, chosen))
    counts = {n: sum(m == n for m, _ in flag_arms) for n, _ in flag_arms}
    seen = {}
    for i, (n, chosen) in enumerate(flag_arms):  # repeated arms are replicates: name-rep1, name-rep2
        if counts[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            flag_arms[i] = (f'{n}-rep{seen[n]}', chosen)
    flag_arms = flag_arms or [(None, [])]
    base_flags = {f for f in ARM_FLAGS if getattr(args, f)}

    def arm_setup(chosen):
        """Set the worker environment and return (controller type, effective flags) for one arm."""
        flags = {f: (f in base_flags or f in chosen) for f in ARM_FLAGS}
        os.environ['FIN_RSI_ALGORITHM_HINT'] = '1' if flags['algorithm_hint'] else '0'
        os.environ['FIN_RSI_SKILL_NAME_HINT'] = '1' if flags['skill_name_hint'] else '0'
        os.environ['FIN_RSI_PERSISTENT_PYTHON'] = '1' if flags['persistent_python'] else '0'
        # Parent-side tools (describe_algorithm, list_skills) use the same class as the worker.
        tools_type = interface.env_tools_class()
        if tools_type is caps.Tools and not (flags['persistent_python'] or flags['python_only_hint']):
            return Controller, flags
        return (type('InterfaceController', (interface.InterfaceControllerMixin, Controller),
                     dict(tools_type=tools_type, persistent_python=flags['persistent_python'],
                          python_only_hint=flags['python_only_hint'])), flags)
    config = dict(vars(args), argv=sys.argv, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  rows_sha256=hashlib.sha256(Path(args.rows).read_bytes()).hexdigest(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'), skill_overlay_file=overlay)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    summary = []
    for spec in map(parse_model, args.model):
        rows = [dict(r, model_tag=spec['tag']) for r in binding['rows']
                if r['condition'] == args.condition and
                r.get('model_tag', spec['tag']) == spec['tag'] and  # no tag: model-agnostic row
                (not wanted or r['unit_id'] in wanted)]
        if not rows:
            raise SystemExit(f"no rows for model {spec['tag']} condition {args.condition}")
        if args.backend == 'hf':
            backend = HFBackend(spec['model_id'], spec['revision'], args.output_cap, SAMPLING[args.sampling])
        elif args.backend.startswith('replay:'):
            backend = ReplayBackend(args.backend.split(':', 1)[1], args.replay_condition)
        else:
            raise SystemExit('unknown backend ' + args.backend)
        seeds = [int(x) for x in args.seeds.split(',') if x]
        rows = [dict(r, sampling_seed=seed) for r in rows for seed in seeds] if seeds else rows
        for task, variant, channel, (arm_name, chosen), row in (
                (t, v, c, a, r) for t in task_variants for v in variants for c in channels
                for a in flag_arms for r in rows):
            controller_type, flags = arm_setup(chosen)
            if channel == 'structured':
                controller_type = type('StructuredController', (interface.DecisionChannelMixin, controller_type), {})
            row_id = f"{row['unit_id']}__{spec['tag']}__{args.interface}__s{row['sampling_seed']}"
            if len(variants) > 1:
                row_id += '__' + variant
            if arm_name is not None:
                row_id += '__' + arm_name.replace('+', '_')
            if task_variants != ['sealed']:
                row_id += '__tv-' + task
            if channel:
                row_id += '__dc-' + channel
            rd = out / 'rows' / row_id
            rd.mkdir(parents=True)
            source = Path(units[row['unit_id']]['PIT_and_input']['visible_workspace'])
            shutil.copytree(source, rd / 'visible')
            backend.start_row(row)
            arm = args.arm or row.get('arm', 'library')
            controller = controller_type(rd, rd / 'visible', arm, menu_seed=row['sampling_seed'])
            if channel == 'structured':
                controller.holdings = dict(units[row['unit_id']]['PIT_and_input']['holdings_before'])
            started = time.monotonic()
            decision = decide(backend, controller, apply_task_variant(row['messages'], task), args.interface,
                              args.max_turns, args.output_cap, variant,
                              args.strip_history_thinking, channel)
            metrics = dict(row_id=row_id, unit_id=row['unit_id'], model=spec['tag'],
                           seed=row['sampling_seed'], arm=arm,
                           interface=args.interface, system_variant=variant,
                           flag_arm=arm_name, skill_search=args.skill_search, skill_overlay=overlay,
                           snapshot_tool_log=args.snapshot_tool_log,
                           **flags,
                           wall_seconds=round(time.monotonic() - started, 1),
                           **row_metrics(decision))
            if task_variants != ['sealed']:
                metrics['task_variant'] = task
            if channel:
                allocs = [c for c in decision['tool_calls'] if c.get('tool') == 'allocate']
                done = [c for c in allocs if c.get('ok')]
                metrics.update(decision_channel=channel, allocate_calls=len(allocs),
                               allocate_errors=sum(1 for c in allocs if not c.get('ok')),
                               submit_redirects=sum(1 for t in decision['turns']
                                                    if interface.SUBMIT_REDIRECT in (t.get('error') or '')),
                               allocation={k: done[-1][k] for k in ('method', 'max_weight', 'max_turnover',
                                                                    'max_weight_achieved', 'turnover_achieved')}
                               if done else None)
            (rd / 'decision.json').write_text(json.dumps(dict(decision, metrics=metrics), indent=1))
            summary.append(metrics)
            print(json.dumps({k: v for k, v in metrics.items() if k != 'algorithm_ids'}), flush=True)
        del backend
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
