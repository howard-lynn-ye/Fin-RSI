"""Check unchanged public P/E inputs with pinned tokenizers; never load model weights."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

try:
    import study_context as context
except ModuleNotFoundError as exc:
    if exc.name != 'study_context':
        raise
    from scripts import study_context as context


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def packets(protocol, public, interface):
    from fin_skills.tasks import frozen_suite
    from fin_skills.rsi import study
    from fin_skills.rsi.run_record import digest
    from benchmarks.agent_study.external_packets import prompt, digest as public_digest
    tasks = frozen_suite().heldout
    if [t.manifest() for t in tasks] != protocol['positions']['evaluation_tasks']:
        raise ValueError('fixed P tasks changed')
    for method in study.METHODS:
        for library in (False,True):
            for task in tasks:
                for seed in protocol['positions']['data_and_generation_seed_ids']:
                    messages = study.initial_messages(task,seed=seed,library=library)
                    yield dict(track='P',method=method.name,library=library,task_id=task.id,seed=seed,
                        task_sha256=task.sha256,frozen_messages_sha256=digest(messages)),messages
    for task in public:
        for library in protocol['external']['experiment']['conditions']:
            messages = prompt(task,interface_text=interface if library else '')
            for seed in protocol['external']['experiment']['seeds']:
                yield dict(track='E',method='external_code',library=library,task_id=task['id'],seed=seed,
                           public_sha256=public_digest(task)),messages


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('context input processing requires a Slurm CPU allocation')
    try:
        import run_frozen_study as runner
    except ModuleNotFoundError as exc:
        if exc.name != 'run_frozen_study':
            raise
        from scripts import run_frozen_study as runner
    runner.assert_runtime_root(args.qualified)
    from transformers import AutoConfig
    from benchmarks.agent_study.transformers_chat import load_tokenizer, tokenize_chat
    from benchmarks.agent_study.rsi_qualify import verify_source
    protocol = read(args.protocol)
    runner.check_context_amendment(args.amendment,context.sha(args.protocol))
    enrollment = read(args.enrollment/'terminal.json')
    if (enrollment['status'] != 'enrollment_frozen'
            or enrollment['protocol_sha256'] != context.sha(args.protocol)):
        raise ValueError('fixed enrollment/protocol required')
    for name in ('public.jsonl','interface.json','episode-plan.json'):
        if context.sha(args.enrollment/name) != enrollment['files'][name]:
            raise ValueError('public input identity changed')
    public = [json.loads(line) for line in (args.enrollment/'public.jsonl').read_text().splitlines()]
    if len(public) != enrollment['selected_questions']:
        raise ValueError('complete fixed external input population required')
    interface = read(args.enrollment/'interface.json')['text']
    all_packets = list(packets(protocol,public,interface))
    inventory = {row['id']:row for row in read(args.inventory)}
    args.output.mkdir(parents=True,exist_ok=False)
    source_sha = verify_source(args.qualified/'source')
    if source_sha != enrollment['qualified_source_manifest_sha256']:
        raise ValueError('qualified numerical source differs from enrollment')
    bound_files = {str(p):context.sha(p) for p in (args.protocol,args.amendment,args.inventory,
        args.enrollment/'terminal.json',Path(__file__),Path(context.__file__))}
    helper_hashes={p.name:context.sha(p) for p in (Path(__file__),Path(context.__file__))}
    write(args.output/'plan.json',dict(protocol_sha256=context.sha(args.protocol),
        amendment_sha256=context.sha(args.amendment),enrollment_terminal_sha256=context.sha(args.enrollment/'terminal.json'),
        source_manifest_sha256=source_sha,model_ids=[m['id'] for m in protocol['models']['twenty_hf_models']],
        helper_sha256=context.sha(__file__),context_helper_sha256=context.sha(context.__file__),
        tokenizer_config_download_permitted=args.allow_tokenizer_download,model_weight_download=False,
        bound_files=bound_files,model_calls=0,publication_ready=False))
    outcomes=[]
    for model in protocol['models']['twenty_hf_models']:
        row=inventory.get(model['id'])
        if row is not None and any(row[k] != model[k] for k in ('model','revision')):
            raise ValueError('inventory differs from fixed model/revision')
        cache=(row['cache_check']['cache'] if row and row.get('cached') else str(args.metadata_cache))
        kwargs=dict(cache_dir=cache,local_files_only=bool(row and row.get('cached')) or not args.allow_tokenizer_download,
                    trust_remote_code=False)
        try:
            # These APIs resolve configuration and tokenizer assets only, never weight files.
            config=AutoConfig.from_pretrained(model['model'],revision=model['revision'],**kwargs)
            tokenizer=load_tokenizer(model['model'],model['revision'],**kwargs)
            native=context.native_identity(config,tokenizer)
            entries=[]; lengths={}
            for key,messages in all_packets:
                packet_sha=context.digest(messages)
                if packet_sha not in lengths:
                    lengths[packet_sha]=len(tokenize_chat(tokenizer,messages)['input_ids'])
                entries.append(dict(key,messages_sha256=packet_sha,prompt_tokens=lengths[packet_sha],output_cap=2048))
            counts=dict(Counter(entry['track'] for entry in entries))
            overflow=sum(row['prompt_tokens']+2048>native['effective_tokens'] for row in entries)
            evidence_name=model['id']+'.initial.jsonl'
            with (args.output/evidence_name).open('x',encoding='utf-8') as stream:
                for entry in entries:stream.write(json.dumps(entry,sort_keys=True)+'\n')
            profile=dict(status='initial_inputs_checked',model_id=model['id'],model=model['model'],revision=model['revision'],
                protocol_sha256=context.sha(args.protocol),amendment_sha256=context.sha(args.amendment),
                enrollment_terminal_sha256=context.sha(args.enrollment/'terminal.json'),source_manifest_sha256=source_sha,
                helper_hashes=helper_hashes,unique_prompt_hash_count=len(lengths),
                native=native,initial_cell_counts=counts,overflow_initial_cells=overflow,
                initial_rows_file=evidence_name,initial_rows_sha256=context.sha(args.output/evidence_name),
                initial_max_prompt_tokens={track:max(r['prompt_tokens'] for r in entries if r['track']==track) for track in counts},
                model_calls=0,metadata_cache=cache,publication_ready=False,
                multi_turn_scope=dict(Repeat='fresh messages each attempt',Check='at most2 prior assistant/feedback pairs',
                    Explain='at most2 prior assistant/feedback pairs; fixed explanations, not S Reflexion',
                    assistant_output_allowance_per_prior_turn=2048,execution_error_feedback_tail_characters=1500,
                    further_checks='complete actual decoded history + unchanged output allowance counted before every provider request',
                    static_universal_history_proof=False,complete_task_fit='only after all actual guarded calls and all fixed cells finish'))
            write(args.output/(model['id']+'.json'),profile)
            outcomes.append(dict(model_id=model['id'],status=profile['status'],effective_tokens=native['effective_tokens'],
                overflow_initial_cells=overflow,profile_sha256=context.sha(args.output/(model['id']+'.json'))))
            del config,tokenizer
        except (OSError,ValueError,RuntimeError) as exc:
            outcome=dict(model_id=model['id'],status='unavailable_or_invalid_metadata',error=f'{type(exc).__name__}: {exc}')
            write(args.output/(model['id']+'.unavailable.json'),outcome)
            outcomes.append(outcome)
    if (any(context.sha(p)!=expected for p,expected in bound_files.items())
            or verify_source(args.qualified/'source') != source_sha):
        raise ValueError('context preflight identity changed during execution')
    terminal=dict(status='complete_metadata_accounting',models=outcomes,model_calls=0,publication_ready=False,
        profile_count=sum(row['status']=='initial_inputs_checked' for row in outcomes),
        scope='Full fixed initial inputs; no gold/scoring/program read and no static claim of complete multi-turn fit')
    write(args.output/'terminal.json',terminal)
    print(json.dumps(terminal,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('qualified','protocol','amendment','enrollment','inventory','metadata-cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--allow-tokenizer-download',action='store_true')
    main(parser.parse_args())
