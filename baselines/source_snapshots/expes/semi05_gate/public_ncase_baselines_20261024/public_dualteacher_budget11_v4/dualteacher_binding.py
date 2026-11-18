"""Own SHA-bound Public split/native clock binding, stdlib only; no admission."""
import ast,copy,dataclasses,hashlib,json,math,os,random,stat
from pathlib import Path
from dualteacher_budget import plan,require
from dualteacher_state import role_window
from dualteacher_native import public_core_ast,CORE_SHA

BY_CONTRACT_SHA='b82d2a57777af9f97925768590eebca718587e44ae40ba073b8751cc41a1a04f'
BUDGET_SHA='9c79216dafe33283c4e49bd64c40e4dbe9c9ff92cff8f0a3b3af02c6c844677b'
OLD_REMOTE='__RECORDED_PROJECT_ROOT__/expes/semi05_gate/dualteacher_fixed_final_20261021_v4_conda/'
NAMES=('budget_protocol.py','runtime_data.py','dualteacher_core.py','dualteacher_backend.py','dualteacher_protocol.py','dualteacher_util.py')

def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def pinned_bytes(path,expected,size=None):
    path=Path(path)
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink source/input')
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size<=8*1024**2,'Small regular source/input required')
        raw=stream.read();after=os.fstat(stream.fileno());named=path.stat()
    ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    require(ident(before)==ident(after)==ident(named) and before.st_ctime_ns==after.st_ctime_ns
        and (os.name=='nt' or after.st_ctime_ns==named.st_ctime_ns),'Source changed during same-FD read')
    require(len(raw)==before.st_size and (size is None or len(raw)==size) and hashlib.sha256(raw).hexdigest()==expected,'Source/input SHA/size differs: '+path.name)
    return raw

def split_ids(raw):
    rows=raw.decode('utf-8').splitlines()
    require(rows and all(x==x.strip() and x.startswith('PUBLIC_NATIVE_PREFIX_') and x[13:].isdigit() for x in rows)
        and len(rows)==len(set(rows)),'Invalid/duplicate case ID')
    return rows

class DualTeacherBinding:
    def __init__(self,labels,*,native_dir,by_contract,budget_file,split_dir):
        self._recipe=plan(labels);origin=json.loads(pinned_bytes(by_contract,BY_CONTRACT_SHA))
        frozen=json.loads(pinned_bytes(budget_file,BUDGET_SHA))
        require((origin['method'],origin['dataset'],origin['fold'],origin['seed'])==('dualteacher_sam','BY',2,2027),'Wrong own native origin')
        require((frozen['dataset'],frozen['fold'],frozen['seed'],frozen['reference_cycles'])==('Public433',2,2027,11)
            and frozen['labels_are_case_counts_not_percent'] is True and frozen['performance_early_stop'] is False,'Frozen cutoff policy changed')
        matches=[m for row in frozen['rows'] if row['labeled_cases']==labels for m in row['method_mapping'] if m['method']=='dualteacher_sam']
        require(len(matches)==1,'Missing method cutoff');m=matches[0]
        require(m['inputs_per_stream_per_iteration']==2 and m['native_ssl_iterations']==self._recipe['ssl_iterations']
            and m['actual_primary_inputs_per_stream']==self._recipe['primary_inputs_per_stream']
            and m['expected_optimizer_updates']==self._recipe['effective_optimizer_updates']
            and m['overshoot_per_stream']==self._recipe['overshoot_per_stream'],'Method cutoff differs')
        raw={};pins={}
        for name in NAMES:
            candidates=[p for p in origin['sources'] if p['path']==OLD_REMOTE+name]
            require(len(candidates)==1,'Missing own source pin: '+name);pin=candidates[0]
            raw[name]=pinned_bytes(Path(native_dir)/name,pin['sha256'],pin['bytes']);pins[name]=copy.deepcopy(pin)
        require(pins['dualteacher_core.py']['sha256']==CORE_SHA,'Own native core differs')
        _,adaptation=public_core_ast(Path(native_dir)/'dualteacher_core.py')
        env=dict(__name__=__name__,dataclass=dataclasses.dataclass,random=random,math=math,need=require,NATIVE_EPOCHS=200,NATIVE_ITERS_PER_EPOCH=200)
        for file,kind,name in [('budget_protocol.py',ast.ClassDef,'Budget'),('runtime_data.py',ast.FunctionDef,'schedule'),('dualteacher_core.py',ast.FunctionDef,'clock')]:
            tree=ast.parse(raw[file]);nodes=[n for n in tree.body if isinstance(n,kind) and n.name==name]
            require(len(nodes)==1,'Native definition closure differs')
            exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),env)
        files={name:split_ids(pinned_bytes(Path(split_dir)/name,digest)) for name,digest in frozen['split_pins'].items()}
        whole,test,prior=(files[n] for n in ('train_whole.txt','test.txt','original_lab17.txt'))
        require((len(whole),len(test),len(prior))==(346,87,17) and not set(whole)&set(test) and set(prior)<set(whole),'Split boundary differs')
        ranked=sorted(prior,key=lambda c:hashlib.sha256(('Public433/fold2/ncase/2027/'+c).encode()).hexdigest())
        for n in (5,10):
            lab,unlab=files['n%03d_lab.txt'%n],files['n%03d_unlab.txt'%n]
            require(lab==ranked[:n] and unlab==[c for c in whole if c not in lab],'Nested selection/order differs')
        self._splits=dict(train_lab=files['n%03d_lab.txt'%labels],train_unlab=files['n%03d_unlab.txt'%labels],test=test)
        self.clock_budget=env['Budget']('Public433',labels,labels,346-labels,cycles=36)
        self._pairs=env['schedule'](self.clock_budget,self._splits)
        require(len(self._pairs)==2*self._recipe['original_clock_iterations'],'Original pair sequence compressed')
        self._clock=env['clock'];self._pins=pins
        self._evidence=dict(status='OWN_SOURCE_SPLIT_CLOCK_BOUND_NOT_GPU_QUALIFIED',by_contract_sha256=BY_CONTRACT_SHA,
            budget_sha256=BUDGET_SHA,native_source_bindings=pins,split_sha256=frozen['split_pins'],recipe=self._recipe,
            cutoff_pairs_sha256=stable(self._pairs[:self._recipe['primary_inputs_per_stream']]),native_adaptation=adaptation,
            foundation_sha256=origin['foundation']['sha256'],patient_files_bound=False,actual_tensor_recovery_verified=False,
            gpu_qualified=False,resource_admitted=False)

    @property
    def recipe(self):return copy.deepcopy(self._recipe)
    @property
    def splits(self):return copy.deepcopy(self._splits)
    @property
    def pairs(self):return copy.deepcopy(self._pairs)
    def evidence(self):return copy.deepcopy(self._evidence)
    def require_core_total(self,total):
        require(type(total) is int and total==self._recipe['original_clock_iterations'],'Native original clock changed')
        return total
    def pair_batch(self,index,role='formal'):
        start,stop=role_window(self._recipe,role)
        require(type(index) is int and start<=index<stop,'Outside native role/cutoff')
        return copy.deepcopy(self._pairs[2*index:2*index+2])
    def original_clock(self,index,role='formal'):
        self.pair_batch(index,role)
        return self._clock(index,self._recipe['original_clock_iterations'])
    def executed_prefix(self,position,role='formal'):
        start,stop=role_window(self._recipe,role)
        require(type(position) is int and start<=position<=stop,'Outside resume role/cutoff')
        return copy.deepcopy(self._pairs[2*start:2*position])
    def native_path(self,name):
        require(name in self._pins,'Unbound native module')
        return Path(self._pins[name]['path'])
