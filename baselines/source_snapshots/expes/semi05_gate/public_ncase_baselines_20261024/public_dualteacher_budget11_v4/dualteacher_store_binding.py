"""Own complete formal-state/ledger closure on injected original-lock IO.
No registry mutation, model construction, test, admission or automatic resume.
"""
import hashlib,os,copy
from pathlib import Path
from dualteacher_storage import RecoveryStore,BoundedSink,safe_file,read_pin
from dualteacher_budget import require,counts
from dualteacher_state import validate_payload
from dualteacher_ledger import scan
class DualTeacherStore(RecoveryStore):
 def __init__(self,job,artifact_root,slots,bound,log_bound,identity,guard,dump,load,binding,foundation,native_recipe,target,terminal_bound,**kwargs):
  self.binding=binding;self.foundation=copy.deepcopy(foundation);self.native_recipe=copy.deepcopy(native_recipe)
  require(type(terminal_bound) is int and terminal_bound>0,'Invalid terminal bound')
  self.target=safe_file(target,artifact_root);require(self.target not in [Path(s).absolute() for s in slots],'Terminal overlaps recovery')
  self.terminal_bound=terminal_bound
  super().__init__(job,artifact_root,slots,bound,log_bound,identity,guard,self.validate_complete_state,dump,load,**kwargs)
 def validate_complete_state(self,state):
  require(state['foundation']==self.foundation,'Own foundation binding differs')
  c=validate_payload(state,self.identity,self.binding.recipe,'formal',self.binding.pairs)
  pin=read_pin(self.job/'formal/steps.jsonl',self.job,True);raw=pin.pop('raw');require(state['ledger']==pin,'State does not bind exact original ledger')
  observed=scan(raw,self.binding);require(observed['partial_bytes']==0 and observed['counts']==c,'Native complete ledger and state counters differ')
  return c
 def validate_terminal(self,state):
  r=self.binding.recipe
  require(set(state)=={'format','method','execution_role','selection','contract_sha256','recipe','counts','signatures','foundation','native_recipe','student','ema','op','criterion'},'Terminal closure differs')
  require(state['format']=='public_dualteacher_fixed11_terminal_v1' and state['method']=='dualteacher_sam' and state['execution_role']=='formal' and state['selection']=='fixed_terminal_EMA_plus_OP','Terminal role/selection differs')
  require(state['contract_sha256']==self.identity and state['recipe']==r and state['counts']==counts(r['ssl_iterations']),'Terminal identity/cutoff differs')
  require(state['foundation']==self.foundation and state['native_recipe']==self.native_recipe,'Terminal native initialization/recipe differs')
  signatures=state['signatures'];require(set(signatures)=={'student','ema','op','criterion'} and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in signatures.values()),'Incomplete terminal signatures')
  observed=scan(read_pin(self.job/'formal/steps.jsonl',self.job,True)['raw'],self.binding)
  require(observed['counts']==state['counts'] and observed['partial_bytes']==0 and observed['complete_reference_cycles']==11 and observed['actual_train_cases']==346,'Incomplete terminal actual ledger')
 def save_terminal(self,state):
  with self.guard.original_lock():
   capacity=self._gate();self.validate_terminal(state);require(not self.target.exists(),'Terminal already exists')
   temporary=safe_file(Path(str(self.target)+'.partial'),self.root);require(not temporary.exists(),'Unreviewed partial terminal')
   self.target.parent.mkdir(parents=True,exist_ok=True)
   with temporary.open('xb') as stream:
    self.dump(state,BoundedSink(stream,self.terminal_bound));stream.flush();os.fsync(stream.fileno())
   pin=read_pin(temporary,self.root);os.link(temporary,self.target);temporary.unlink();self.sync_directory(self.target.parent)
   return dict(path=str(self.target),bytes=pin['bytes'],sha256=pin['sha256'],resource_lock_verified=True,capacity_before=capacity)
 def load_terminal(self,artifact):
  with self.guard.original_lock():
   self._gate();require(artifact['path']==str(self.target),'Different terminal target')
   with self.target.open('rb') as stream:
    before=os.fstat(stream.fileno());digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    require(before.st_size==artifact['bytes'] and digest.hexdigest()==artifact['sha256'],'Terminal SHA/size differs')
    stream.seek(0);value=self.load(stream);after=os.fstat(stream.fileno());named=self.target.stat()
    ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink)
    require(ident(before)==ident(after)==ident(named) and before.st_nlink==1,'Terminal changed during same-fd load')
   self.validate_terminal(value);return value
