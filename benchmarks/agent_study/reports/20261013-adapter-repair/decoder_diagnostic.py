"""Bounded decoder diagnostics; never used as a trading decision."""
import json
from pathlib import Path
import os
import subprocess, sys
import torch
from transformers import LogitsProcessor
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as preflight
from benchmarks.agent_study.transformers_chat import TransformersChat
B=Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
response=B/'native-queue-20261013-v1/8092cb46ede240d83027fe1de82ece898cecf160aa3445fd3511c99bec1a4cc6/response.json'
r=json.loads(response.read_text())
print(json.dumps({'native_failure': {k:r.get(k) for k in ('success','exit_code','error','requested_model','reported_models')}, 'tool_events':[{k:e.get(k) for k in ('event','type')}|{'step_type':e.get('step_update',{}).get('step_type')} for e in r.get('native_tool_events',[])], 'results':[e.get('result',{}) for e in r.get('events',[]) if e.get('event')=='result'], 'stderr':r.get('stderr','')[:1800]},ensure_ascii=False),flush=True)

subprocess.run([sys.executable, '/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-context-inspect-deepseek-template.py'], env=dict(os.environ, HF_HUB_CACHE=str(B/'matrix-weights-v2/cache/hf/hub')), check=True)
os.environ['HF_HUB_CACHE']='/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub'
root=B/'v9-access-matrix-20261013-r4/pairs/14b-11'
p=study.verify(root)
print(json.dumps({'gpu':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'model':p['model']}),flush=True)
b=TransformersChat(p['model'][1],p['model'][2],max_tokens=p['max_tokens'],seed=p['seed']*1000)
print(json.dumps({'generation_config':b.model.generation_config.to_dict()}),flush=True)
original=b.model.generate
class Observe(LogitsProcessor):
    def __call__(self, ids, scores):
        finite=torch.isfinite(scores)
        bad=torch.isnan(scores).any() or torch.isposinf(scores).any() or not finite.any(dim=-1).all()
        if bad:
            print(json.dumps({'post_processor_failure':{'length':ids.shape[-1], 'nan':int(torch.isnan(scores).sum()), 'positive_inf':int(torch.isposinf(scores).sum()), 'finite':int(finite.sum()), 'tail_ids':ids[0,-50:].tolist(), 'tail':b.tokenizer.decode(ids[0,-80:])}}),flush=True)
            raise RuntimeError('Diagnostic stopped before invalid multinomial')
        return scores
steps=0
def observe_forward(module,args,output):
    global steps
    last=output.logits[:,-1,:]
    if steps==0 or not torch.isfinite(last).all():
        print(json.dumps({'forward':{'step':steps,'finite':int(torch.isfinite(last).sum()),'total':last.numel(),'nan':int(torch.isnan(last).sum()),'min':float(last.min()),'max':float(last.max())}}),flush=True)
    steps+=1
b.model.register_forward_hook(observe_forward)
def generate(*args,**kwargs):
    return original(*args,**kwargs,logits_processor=[Observe()])
b.model.generate=generate
try:
    result=b.final_decision(preflight.decoder_messages(root,p))
    print(json.dumps({'success':result}),flush=True)
except Exception as e:
    print(json.dumps({'diagnostic_error':type(e).__name__+': '+str(e),'steps':steps}),flush=True)
    raise