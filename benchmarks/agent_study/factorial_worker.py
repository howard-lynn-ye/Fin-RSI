"""Confined agent code; package calls occur only through logged study helpers."""
import builtins,contextlib,importlib,io,json,os,sys,traceback
from pathlib import Path
from benchmarks.agent_study import factorial_tools as ft

def main():
    box,workspace=(Path(p).resolve() for p in sys.argv[1:3]);req=json.loads((box/'request.json').read_text())
    import numpy,pandas,scipy.optimize,scipy.cluster.hierarchy
    import fin_skills,fin_skills.algorithms
    from benchmarks.agent_study import trading_tools_v5,trading_worker,trading_study
    from benchmarks.agent_study.linux_sandbox import confine
    corpus=json.loads((box/'corpus.json').read_text()) if req['arm'] in ft.KNOWLEDGE else {}
    package=Path(fin_skills.__file__).parent
    paths=[Path(sys.prefix),Path(sys.base_prefix),Path('/usr'),Path('/lib'),Path('/lib64'),workspace]
    if os.environ.get('FIN_STUDY_SHARED_RUNTIME'):paths.append(Path(os.environ['FIN_STUDY_SHARED_RUNTIME']))
    if req['arm'] in ft.NUMERICAL:paths += [p for p in package.iterdir() if p.name not in ('_skills','__pycache__')]
    for p in paths:
        if Path(__file__).resolve().parent.is_relative_to(p.resolve()):raise RuntimeError('read grant exposes study source')
    actual_import=builtins.__import__;actual_module=importlib.import_module
    def filtered(name,*args,**kwargs):
        if (name=='fin_skills' or name.startswith('fin_skills.')) and not ft.ALLOW_INTERNAL_IMPORT:raise ImportError('use the recorded bound study functions; direct package imports are disabled in every condition')
        return actual_import(name,*args,**kwargs)
    def filtered_module(name,*args,**kwargs):
        if (name=='fin_skills' or name.startswith('fin_skills.')) and not ft.ALLOW_INTERNAL_IMPORT:raise ImportError('direct package imports disabled')
        return actual_module(name,*args,**kwargs)
    builtins.__import__=filtered;importlib.import_module=filtered_module
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    os.chdir(workspace);confine(box,paths)
    tools=ft.Tools(workspace,req['arm'],req['tickers'],req['menu_seed'],corpus)
    if req['tool']!='run_python':result=tools.call(req['tool'],req['arguments'])
    else:
        out,err,target=io.StringIO(),io.StringIO(),None
        class Submitted(BaseException):pass
        def submit(weights):
            nonlocal target
            target,problem=ft.validate(weights,req['tickers'])
            if problem:raise ValueError(problem)
            raise Submitted()
        ok,error=True,None
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:exec(compile(req['arguments'].get('code',''),'agent_code.py','exec'),dict(__name__='__main__',submit=submit,**tools.python_bindings()))
            except Submitted:pass
            except BaseException as exc:
                ok=False;target=None;error=f'{type(exc).__name__}: {str(exc)[:500]}';traceback.print_exc(limit=4)
        result=dict(ok=ok,error=error,output=out.getvalue()+('\n[stderr]\n'+err.getvalue() if err.getvalue() else ''),submission=target,tool_calls=tools.calls)
    (box/'result.json').write_text(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
