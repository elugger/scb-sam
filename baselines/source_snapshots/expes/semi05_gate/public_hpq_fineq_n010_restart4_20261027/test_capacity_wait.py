import contextlib
import json
import random
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import capacity_wait as w

def shortage():
    return RuntimeError(w.PREFIX + json.dumps(dict(margin_bytes=-1, free_bytes=9, required_bytes=10)))

class CapacityWaitTests(unittest.TestCase):
    def make(self, fail=1):
        self.depth=0;self.calls=0
        @contextlib.contextmanager
        def lock():
            self.depth+=1
            try:yield
            finally:self.depth-=1
        def capacity(n):
            self.assertEqual(self.depth,1);self.assertEqual(n,123)
            self.calls+=1
            if self.calls<=fail:raise shortage()
            return {'passed':True}
        return types.SimpleNamespace(original_lock=lock,capacity_snapshot=capacity)

    def test_repeated_rejections_release_lock_then_admit_once(self):
        g=self.make(3);slept=[];bodies=[]
        def sleep(n):self.assertEqual(self.depth,0);slept.append(n)
        with w.admitted_capacity(g,123,sleep=sleep) as c:
            self.assertEqual(self.depth,1);bodies.append(c)
        self.assertEqual(slept,[60]*3);self.assertEqual(bodies,[{'passed':True}]);self.assertEqual(self.depth,0)

    def test_live_parameters_optimizer_rng_and_next_step_identical(self):
        torch.manual_seed(2027);random.seed(2027)
        model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.ones(1,2)
        model(x).sum().backward();optimizer.step();optimizer.zero_grad()
        params=[p.detach().clone() for p in model.parameters()]
        moments=[v.detach().clone() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
        rng=torch.get_rng_state().clone();py=random.getstate()
        def sleep(n):
            self.assertEqual(self.depth,0)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(random.getstate(),py)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(params,model.parameters())))
            now=[v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(moments,now)))
        import copy
        reference=copy.deepcopy(model);ref_opt=torch.optim.Adam(reference.parameters(),lr=.01)
        ref_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        with w.admitted_capacity(self.make(2),123,sleep=sleep):pass
        for m,o in ((model,optimizer),(reference,ref_opt)):
            m(x).sum().backward();o.step()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),reference.parameters())))

    def test_error_from_body_never_retried(self):
        g=self.make(0)
        with self.assertRaises(RuntimeError):
            with w.admitted_capacity(g,123,sleep=lambda n:self.fail('Body retry')):raise shortage()
        self.assertEqual(self.calls,1)

    def test_ram_identity_partial_errors_not_waited(self):
        for message in ('Insufficient RAM','Existing partial','Changed source'):
            g=self.make(0);g.capacity_snapshot=lambda n:(_ for _ in ()).throw(RuntimeError(message))
            with self.assertRaisesRegex(RuntimeError,message):
                with w.admitted_capacity(g,123,sleep=lambda n:self.fail('Unsafe retry')):pass

    def test_safety_precedes_best_and_no_wait_inside_training(self):
        import ast
        tree=ast.parse(Path(__file__).with_name('selection_runner.py').read_text())
        calls=sorted((n.lineno,n.func.id) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name))
        names=[x[1] for x in calls]
        self.assertLess(names.index('save_epoch'),names.index('save_best'))

if __name__=='__main__':unittest.main()
