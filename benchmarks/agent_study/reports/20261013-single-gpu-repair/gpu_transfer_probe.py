"""Tiny GPU transfer diagnostic; no model weights, market data or trading."""
import json, os, torch
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'visible':os.environ.get('CUDA_VISIBLE_DEVICES')}),flush=True)
for src,dst in ((0,1),(1,0)):
    if torch.cuda.device_count()<2:
        raise RuntimeError('Two allocated GPUs required')
    reference=torch.linspace(-20,20,152064,device=f'cuda:{src}',dtype=torch.bfloat16).reshape(1,1,-1)
    expected=reference.float().cpu()
    for mode in ('combined_copy_cast','cast_then_copy','host_bounce'):
        if mode=='combined_copy_cast':
            actual=reference[:,-1,:].to(copy=True,dtype=torch.float32,device=f'cuda:{dst}')
        elif mode=='cast_then_copy':
            actual=reference[:,-1,:].float().to(device=f'cuda:{dst}',copy=True)
        else:
            actual=reference[:,-1,:].float().cpu().to(device=f'cuda:{dst}')
        got=actual.cpu()
        print(json.dumps({'src':src,'dst':dst,'peer':torch.cuda.can_device_access_peer(src,dst),'mode':mode,'nan':int(torch.isnan(got).sum()),'inf':int(torch.isinf(got).sum()),'exact':torch.equal(got,expected[:,0,:]),'max_error':float(torch.nan_to_num(got-expected[:,0,:]).abs().max())}),flush=True)