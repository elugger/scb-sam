"""A0B: frozen matrix training plus bounded, qualification-only component checks.

Checks replay the SAME components as the old restore route at cold/warm states.
They do not assert bitwise equivalence of differently ordered full updates.
No qualification probe is called during formal training.
"""
import torch
import torch.nn.functional as F
from expes.semi05_gate.paired_fcd_matrix_20261006 import MatrixHalf
from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
from expes.semi05_gate.paired_r1_restore_full_20261005 import PairedR1RestoreFull
from expes.semi05_gate.paired_hard_relations_20261005 import PairedHardRelations
from expes.semi05_gate.fcd_hard_spatial_relations_20261004 import pick_regions, reliable_class_loss
from expes.semi05_gate.fcd_cac_fg_20260928 import context_views
from expes.semi05_gate.fcd_dycon_r2_losses_20260927 import _image_signal


class QualifiedBalancedHalf(MatrixHalf):
    def _original_component_check(self, item, parameters):
        epoch, step = item['epoch'], item['step']
        # The loss source called here is exactly the old restore original path.
        with self.isolated_context(epoch, step):
            total, sup, pseudo, wsv, logits, stats, views = PairedCrossScaleHalf._losses(
                self, item['image_l'], item['label_l'], item['image_u'], epoch, step)
            values = dict(supervised=float(sup.detach()), pseudo=float(pseudo.detach()),
                          wsv=float(wsv.detach()), views=views)
            grads = torch.autograd.grad(self.scaler.scale((pseudo+wsv)/5.), parameters, allow_unused=True)
            del total, sup, pseudo, wsv, logits
        previous = getattr(self, 'restore_arm', None)
        self.restore_arm = 'restore'
        try:
            old_grads, old = PairedR1RestoreFull.original_gradients(self, item, parameters)
        finally:
            if previous is None: del self.restore_arm
            else: self.restore_arm = previous
        assert values['pseudo'] == old['pseudo'] and values['wsv'] == old['wsv']
        assert values['supervised'] == old['supervised_discarded']
        assert values['views'] == old['view_state']
        assert old['supervised_added'] == 0. and old['declared_state_restored']
        assert all((x is None and y is None) or
                   (x is not None and y is not None and torch.equal(x, y))
                   for x,y in zip(grads, old_grads)), 'Original HALF unlabeled component differs from restore'
        del grads, old_grads
        with self.isolated_context(epoch, step):
            sup = self._supervised_step(item['image_l'], item['label_l'])
            assert float(sup.detach()) == values['supervised'], 'Supervised component differs'
        return dict(epoch=epoch, step=step, **values, original_unlabeled_gradients_exact=True,
                    supervised_component_exact=True, supervised_added=0.,
                    declared_state_restored=True, comparison='components only; not reordered total updates')

    def contextual_gradients(self, image, epoch, step, parameters):
        grads, row = super().contextual_gradients(image, epoch, step, parameters)
        if not getattr(self, '_checking_context_component', False) or step % 5:
            return grads, row
        # Original restore uses feature-capture forwarding and pick_regions.
        # Replaying this route must preserve the new route's values AND gradients.
        with self.isolated_context(epoch, step):
            axis=(epoch+step)%3; a,b=context_views(image,axis)
            with torch.no_grad():
                ta,_=PairedHardRelations.relation_forward(self,self.teacher,a)
                tb,_=PairedHardRelations.relation_forward(self,self.teacher,b)
                qa=F.avg_pool3d(ta.float().softmax(1)[:,1:2],4,4)
                qb=F.avg_pool3d(tb.float().softmax(1)[:,1:2],4,4)
            _,_,reliable,_=pick_regions(qa,qb,_image_signal(image,(32,32,32)),axis)
            logits,_=PairedHardRelations.relation_forward(self,self.student,a)
            objective=reliable_class_loss(logits,qa,qb,reliable)
            assert float(objective.detach()) == row['balanced'], 'Restore CE value differs'
            old_grads=torch.autograd.grad(self.scaler.scale(objective/5.),parameters,allow_unused=True)
            assert all((x is None and y is None) or
                       (x is not None and y is not None and torch.equal(x,y))
                       for x,y in zip(grads,old_grads)), 'Restore CE gradient differs'
            target=F.interpolate((qb>=.5).float(),size=(128,)*3,mode='nearest').long()[:,0]
            mask=F.interpolate(reliable.float(),size=(128,)*3,mode='nearest').bool()[:,0]
            counts=[int((mask&(target==c)).sum()) for c in (0,1)]
            assert counts == [row['background_voxels'],row['foreground_voxels']]
            assert int(reliable.sum())==row['reliable_blocks']
        del old_grads
        row['old_restore_ce_component_exact']=True
        self._restore_context_rows.append(dict(epoch=epoch,step=step,counts=counts,
                                              value=row['balanced'],value_and_gradients_exact=True))
        return grads,row

    def step_matrix(self, raw, compute_context=True, inject_overflow=False):
        if not hasattr(self,'_restore_component_rows'):
            self._restore_component_rows=[]; self._restore_context_rows=[]
        check=(getattr(self,'qualification_component_checks',False) and self.matrix_c=='B'
               and compute_context and not inject_overflow and len(self._restore_component_rows)<2)
        if check:
            for name,module in self.roots().items(): module.train(not name.startswith('teacher'))
            self.optimizer.zero_grad(set_to_none=True); self.clear_relation_graph()
            parameters=[p for group in self.optimizer.param_groups for p in group['params']]
            self._restore_component_rows.append(self._original_component_check(raw[0],parameters))
        self._checking_context_component=check
        try: return super().step_matrix(raw,compute_context,inject_overflow)
        finally: self._checking_context_component=False

    def component_check_report(self):
        assert len(self._restore_component_rows)==len(self._restore_context_rows)==2
        assert all(r['original_unlabeled_gradients_exact'] and r['supervised_component_exact']
                   for r in self._restore_component_rows)
        return dict(status='COMPONENTS_EXACT_ACCUMULATION_NOT_CLAIMED',
                    original_rows=self._restore_component_rows,context_rows=self._restore_context_rows,
                    optimizer_updates_added_by_probes=0,test_forwards_added_by_probes=0,
                    limitation='Two finite cold/warm component checks; old restore total updates retain different summation order and zero-relation computation. No full-update equivalence claim.')
