"""One local readout-feature write; all later generation remains native."""
from __future__ import annotations
import hashlib
import numpy as np
import native_adapter as native
from native_readout import compare,pair_values
from trace_core import group_mask


def selected_values(x,pairs):
    f,g=pairs[:,0].tolist(),pairs[:,1].tolist()
    return np.stack([native.array(x[0,f,g]),native.array(x[0,g,f])],axis=1)


def modify_features(x,previous,pairs,groups,d_message,kind,key,torch):
    """Per-group norm-matched orthogonal controls; local RNG only.

    Geometry controls preserve the equality of the two directed copies. Random
    controls are computational feature perturbations, not guaranteed physical
    coordinate changes. This also applies to mixed temporal feature states.
    """
    native.require(x.shape==previous.shape and x.dtype==previous.dtype and x.dtype==torch.float32,'Head input identity/dtype differs')
    current=selected_values(x,pairs);old=selected_values(previous,pairs)
    layout=group_mask(d_message,x.shape[-1]);used=current.copy();details=[]
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    for group in groups:
        mask=np.broadcast_to(layout==group,current.shape)
        delta=np.where(mask,old.astype(float)-current.astype(float),0.)
        length=float(np.linalg.norm(delta))
        informative=length>10*np.finfo(np.float32).eps*max(1.,float(np.linalg.norm(np.where(mask,current,0.))))
        if kind=='targeted':
            used[mask]=old[mask]
        elif kind=='random_direction' and length>0:
            noise=rng.normal(size=current.shape)
            if group==2:
                native.require(np.array_equal(delta[:,0],delta[:,1]),'Geometry reciprocal features differ')
                noise[:]=(0.5*(noise[:,0]+noise[:,1]))[:,None,:]
            noise=np.where(mask,noise,0.)
            noise-=float(np.sum(noise*delta))/(length*length)*delta
            norm=float(np.linalg.norm(noise));native.require(norm>1e-12,'Degenerate orthogonal random direction')
            proposal=current.astype(float)+noise*(length/norm)
            used[mask]=proposal[mask].astype(current.dtype)
        elif kind!='random_direction':
            native.require(kind=='sham','Unknown intervention kind')
        actual=np.where(mask,used.astype(float)-current.astype(float),0.)
        actual_norm=float(np.linalg.norm(actual))
        cosine=float(np.sum(actual*delta)/(actual_norm*length)) if actual_norm>0 and length>0 else None
        matched=bool(abs(actual_norm-length)<=1e-6+1e-3*length)
        if kind!='sham' and informative:native.require(matched,'Control magnitude differs by more than declared numerical tolerance')
        if kind=='random_direction' and informative:native.require(cosine is not None and abs(cosine)<1e-3,'Random direction is not orthogonal after float conversion')
        details.append(dict(group=int(group),target_norm=length,actual_norm=actual_norm,
                            cosine_to_target=cosine,magnitude_informative=informative,magnitude_matched=matched))
    result=x.detach().clone(memory_format=torch.preserve_format)
    f,g=pairs[:,0].tolist(),pairs[:,1].tolist()
    result[0,f,g]=torch.as_tensor(used[:,0],dtype=x.dtype,device=x.device)
    result[0,g,f]=torch.as_tensor(used[:,1],dtype=x.dtype,device=x.device)
    allowed=torch.zeros_like(x,dtype=torch.bool)
    selected=np.isin(layout,groups)
    allowed[0,f,g]=torch.as_tensor(selected[0],device=x.device)
    allowed[0,g,f]=torch.as_tensor(selected[1],device=x.device)
    native.require(torch.equal(result[~allowed],x[~allowed]),'Unselected feature entries changed')
    return result,dict(groups=groups,per_group=details,only_selected_entries_changed=True,
                       applied=not torch.equal(result,x),total_norm=float(np.linalg.norm(used.astype(float)-current.astype(float))),
                       random_control_seed=seed if kind=='random_direction' else None)


class HeadPatch:
    def __init__(self,model,trial,clock,torch,data,source=None,expected=None):
        self.model,self.trial,self.clock,self.torch=model,trial,clock,torch
        self.data,self.source,self.expected=data,source,expected
        self.pairs=data['pairs'];self.capture={};self.receipt=None;self.visits=0;self.post_visits=0
        self.d_message=model.gen.ligand_dec.bond_refine.node_proj.out_features
        self.handles=[]
        try:
            self.handles.append(model.gen.ligand_dec.bond_refine.message_mlp.register_forward_pre_hook(self.pre))
            self.handles.append(model.register_forward_hook(self.post,with_kwargs=True))
        except BaseException:
            self.close();raise

    def pre(self,module,args):
        k=self.clock['call'];t=self.trial;x=args[0];torch=self.torch
        native.require(x.ndim==4 and x.shape[0]==1,'Expected full native head shape')
        if k==t['call']-1:
            native.require('before' not in self.capture,'Multiple head visits per network call')
            self.capture['before']=x.detach().cpu().clone()
        if k!=t['call']:return None
        self.visits+=1;native.require(self.visits==1 and 'before' in self.capture,'Missing/duplicated intervention visit')
        self.capture['after']=x.detach().cpu().clone();previous=self.capture['before'].to(x.device)
        matches={}
        if self.source is not None:
            for name in ['before','after']:
                matches[name]=compare(self.capture[name],self.source[name])
                native.require(matches[name]['passed'],f'{name} head features differ from the same-seed sham')
        if t['kind']=='sham':
            self.receipt=dict(call=k,kind='sham',family='none',applied=False,total_norm=0.,only_selected_entries_changed=True,per_group=[])
            return None
        key=f"{t['control_seed']}:{t['seed']}:{k}:{t['family']}:{t.get('control_index',-1)}"
        changed,info=modify_features(x,previous,self.pairs,t['groups'],self.d_message,t['kind'],key,torch)
        self.receipt=dict(call=k,kind=t['kind'],family=t['family'],same_seed_source_checks=matches,**info)
        self.capture['modified_selected']=torch.from_numpy(selected_values(changed,self.pairs))
        return (changed,*args[1:])

    def post(self,module,args,kwargs,out):
        if self.clock['call']!=self.trial['call']:return None
        self.post_visits+=1;native.require(self.receipt is not None,'Head intervention did not run')
        original=self.data['calls'][self.trial['call']]['network_output_before_postprocessing']
        for name,value in out.items():
            if name!='bonds':native.compare_tree(value,original[name],self.torch,'output.'+name)
        self.receipt['nonbond_outputs_unchanged_at_intervention']=True
        self.capture['used_pair_logits']=pair_values(out['bonds'],self.pairs).detach().cpu().clone()
        if self.expected is not None:
            native.require(np.array_equal(self.pairs,self.expected['pairs']),'Boundary pair selection differs from V2')
            c=15 if self.trial['kind']=='sham' else 15-sum(1<<g for g in self.trial['groups'])
            target=self.torch.as_tensor(self.expected['logits'][c],dtype=out['bonds'].dtype)
            check=compare(self.capture['used_pair_logits'],target,categorical=True)
            self.receipt['previous_local_experiment_match']=check
            native.require(check['passed'],'Actual rollout intervention does not reproduce the already observed local effect')
        return None

    def validate_complete(self):
        native.require(self.visits==1 and self.post_visits==1 and self.receipt is not None,'Intervention was not executed exactly once')
        if self.trial['kind']=='sham':native.require(not self.receipt['applied'],'Sham changed head input')

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
