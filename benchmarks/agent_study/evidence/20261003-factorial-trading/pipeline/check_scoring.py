"""Synthetic completed-batch fixture, never a model result or paper measurement."""
import tempfile,shutil,json
from pathlib import Path
from benchmarks.agent_study import factorial_study as fs,trading_study as old
from benchmarks.agent_study.factorial_tools import ARMS

def check(root):
    with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
        temp=Path(td);(temp/'datasets').mkdir()
        (temp/'datasets/original').symlink_to(root/'datasets/original',target_is_directory=True)
        for file in ('PROTOCOL.md','corpus.json'):(temp/file).symlink_to(root/file)
        pair=temp/'batch/original-7b-11';fs.freeze(pair,'original','7b',11)
        p=old.load(pair/'protocol.json');inp=old.load(pair/'inputs.json');weights=dict.fromkeys(p['tickers'],1/len(p['tickers']));receipts={}
        for arm in ARMS:
            for i,pick in enumerate(inp['decision_sessions']):
                name=f'decisions/{arm}/{i:02d}.json';old.write(pair/name,dict(target=weights,submitted=True,turns=[],tool_calls=[]));receipts[name]=old.sha(pair/name)
        old.write(pair/'inference-receipt.json',dict(protocol_sha256=old.sha(pair/'protocol.json'),decisions=receipts))
        result=fs.score(pair)
        assert all(abs(x)<1e-12 for x in result['effects_pp'].values())
        assert len({v['return_rate_pct'] for v in result['paths'].values()})==1
        for arm,v in result['paths'].items():
            assert v['submitted']==len(inp['decision_sessions'])
            assert v['fixed_decision_cost_sensitivity_pct']['0']>v['return_rate_pct']>v['fixed_decision_cost_sensitivity_pct']['25']
        # Corrupt a decision and require receipt validation to refuse the batch.
        target=pair/'decisions/base/00.json';target.write_text(target.read_text()+' ')
        refused=False
        try:fs.score(pair)
        except AssertionError:refused=True
        assert refused,'receipt must reject even whitespace changes in a frozen decision'
        return dict(passed=True,label='synthetic equal-weight fixture; no LLM outputs',same_targets_zero_factor_effects=True,cost_ordering=True,corrupt_receipt_rejected=True)
