"""Isolated NumPy tensor adapter for software tests; not a pretrained model."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib.util,json,os,random,types,sys
import numpy as np
ROOT=Path.cwd()
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import native_adapter as m
helper=m.import_file(PACKAGE/'adapters/replay.py','feedback_test_helpers')
def u(x):return x.x if isinstance(x,T) else x
class T:
    def __init__(self,x,dtype=None):self.x=np.asarray(u(x),dtype=dtype)
    @property
    def shape(self):return self.x.shape
    @property
    def ndim(self):return self.x.ndim
    @property
    def dtype(self):return self.x.dtype
    @property
    def device(self):return types.SimpleNamespace(type='cpu')
    def size(self,d=None):return self.shape if d is None else self.shape[d]
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return T(self.x[i])
    def __setitem__(self,i,v):self.x[i]=u(v)
    def __add__(self,v):return T(self.x+u(v))
    def __radd__(self,v):return T(u(v)+self.x)
    def __sub__(self,v):return T(self.x-u(v))
    def __rsub__(self,v):return T(u(v)-self.x)
    def __mul__(self,v):return T(self.x*u(v))
    def __rmul__(self,v):return T(u(v)*self.x)
    def __truediv__(self,v):return T(self.x/u(v))
    def __ge__(self,v):return bool(np.all(self.x>=u(v)))
    def __le__(self,v):return bool(np.all(self.x<=u(v)))
    def __float__(self):return float(self.x.reshape(-1)[0])
    def clone(self):return T(self.x.copy())
    def detach(self):return self
    def cpu(self):return self
    def to(self,*args,**kwargs):return self
    def numpy(self):return self.x
    def float(self):return T(self.x.astype(np.float32))
    def long(self):return T(self.x.astype(np.int64))
    def bool(self):return T(self.x.astype(bool))
    def argmax(self,axis=-1):return T(self.x.argmax(axis))
    def tolist(self):return self.x.tolist()
    def unsqueeze(self,d):return T(np.expand_dims(self.x,d))

class Torch:
    state=T(np.zeros(4,np.uint8))
    is_tensor=staticmethod(lambda x:isinstance(x,T))
    linspace=staticmethod(lambda a,b,n:T(np.linspace(a,b,n,dtype=np.float32)))
    zeros_like=staticmethod(lambda x:T(np.zeros_like(u(x))))
    zeros=staticmethod(lambda *shape,**kw:T(np.zeros(shape,np.float32)))
    argmax=staticmethod(lambda x,dim:T(np.argmax(u(x),axis=dim)))
    no_grad=staticmethod(contextlib.nullcontext)
    as_tensor=staticmethod(lambda x,device=None,dtype=None:T(x,dtype=dtype))
    equal=staticmethod(lambda a,b:np.array_equal(u(a),u(b)))
    @classmethod
    def get_rng_state(cls):return cls.state.clone()
    @classmethod
    def set_rng_state(cls,x):cls.state=x.clone()

def softmax(x,dim=-1):
    a=u(x);e=np.exp(a-a.max(axis=dim,keepdims=True));return T(e/e.sum(axis=dim,keepdims=True))

def snap(x):
    if isinstance(x,T):return x.clone()
    if isinstance(x,np.ndarray):return T(x.copy())
    if isinstance(x,dict):return {k:snap(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(snap(v) for v in x)
    return copy.deepcopy(x)

def rng_state():return {'python':random.getstate(),'numpy':snap(np.random.get_state()),'torch_cpu':Torch.get_rng_state()}

class Handle:
    def __init__(self,store,key):self.store,self.key=store,key
    def remove(self):self.store.pop(self.key,None)

class Mod:
    def __init__(self,fn=None):self.fn=fn;self.pre={};self.post={};self.count=0
    def register_forward_pre_hook(self,fn,with_kwargs=False):
        self.count+=1;self.pre[self.count]=(fn,with_kwargs);return Handle(self.pre,self.count)
    def register_forward_hook(self,fn,with_kwargs=False):
        self.count+=1;self.post[self.count]=(fn,with_kwargs);return Handle(self.post,self.count)
    def __call__(self,*args,**kw):
        for fn,k in list(self.pre.values()):
            result=fn(self,args,kw) if k else fn(self,args)
            if result is not None:
                args,kw=result
        value=self.forward(*args,**kw) if self.fn is None else self.fn(*args,**kw)
        for fn,k in list(self.post.values()):
            update=fn(self,args,kw,value) if k else fn(self,args,value)
            if update is not None:value=update
        return value

class Vocab:
    def __init__(self,items):self.items=items
    def tokens_from_indices(self,ids):return [self.items[int(i)] for i in ids]

class Builder:
    vocab=Vocab(['<PAD>','C']);vocab_charges=Vocab([0,1])
    def inpaint_molecule(self,data,prediction,**kw):
        for k in ['coords','atomics','charges']:
            if k in prediction:prediction[k].x[:,0]=data[k].x[:,0]
        return prediction
    def undo_zero_com_batch(self,coords,mask,com_list):return (coords+com_list[0])*mask.unsqueeze(-1)
    def write_xyz_file_from_batch(self,**kw):pass
    def add_ligand_to_pocket(self,lig_data,pocket_data):return lig_data

class Integrator:
    use_sde_simulation=False
    def step(self,curr,pred,prior,times,dt):
        out={k:snap(v) for k,v in curr.items()}
        out['coords']=curr['coords']+dt*(pred['coords']-curr['coords'])
        for k in ['atomics','bonds','charges']:out[k]=pred[k].clone()
        return out

class Gen:
    def __init__(self,d):
        def attention(e):return T(u(e)*np.float32(.13)),T(np.zeros((1,6,2),np.float32)),T(np.zeros((1,6,6,2),np.float32))
        self.ligand_dec=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attention=Mod(attention),
             cond_attention=Mod(attention),equi_ff=Mod(lambda e:T(u(e)*np.float32(.09))))])
        self.d=d
    def get_pocket_encoding(self,*a,**kw):return T(np.random.randn(1,4,3,self.d).astype(np.float32)),T(np.ones((1,4,2),np.float32))

class Model(Mod):
    def __init__(self,d=5):
        super().__init__();self.gen=Gen(d);self.builder=Builder();self.integrator=Integrator();self.coord_scale=1.6
        self.sc_charges=False;self._inpaint_self_condition=True;self.inpainting_mode=True;self.inpainting_mode_inf='fragment'
        self.graph_inpainting=False;self.self_condition=True;self.sc_recycle_steps=1;self.flow_interactions=False
        self.predict_interactions=False;self.self_condition_mode='standard';self.device=types.SimpleNamespace(type='cpu')
        self.hparams=types.SimpleNamespace(save_dir='synthetic_no_files');self.recorded_writes=[]
    def forward(self,curr,pocket,t,cond_batch=None,**kw):
        e=np.repeat(u(curr['coords'])[...,None],self.gen.d,axis=-1).astype(np.float32)
        e+=np.float32(.1)*np.repeat(u(cond_batch['coords'])[...,None],self.gen.d,axis=-1)
        e+=np.random.randn(*e.shape).astype(np.float32)*np.float32(.01)
        eq,_,_=self.gen.ligand_dec.layers[0].self_attention(T(e))
        self.recorded_writes.append(m.array(eq[0]))
        coords=curr['coords']+T(u(eq).mean(-1)*np.float32(.3)+np.float32(.02))
        return {'coords':coords,'atomics':T(u(curr['atomics'])*np.float32(3)),
                'bonds':T(u(curr['bonds'])*np.float32(2)),'charges':T(u(curr['charges'])*np.float32(3)),
                'mask':curr['mask'].clone()}
    def _save_trajectory(self,*a,**kw):pass

tree=ast.parse((ROOT/'flowr/models/fm_pocket.py').read_text())
cls=next(v for v in tree.body if isinstance(v,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='_generate' for f in v.body))
funcs=[v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name in {'_generate','_update_times','_get_predictions','_build_trajectory'}]
ns={'torch':Torch,'F':types.SimpleNamespace(softmax=softmax),'np':np,'os':os,'raise_if_cancelled':lambda *a:None}
exec(compile(ast.Module(body=funcs,type_ignores=[]),'actual_ROOT_sampler','exec'),ns)
for f in funcs:setattr(Model,f.name,ns[f.name])

def fixture(d=5):
    random.seed(515);np.random.seed(515);model=Model(d);n=6
    prior={'coords':T(np.arange(n*3,dtype=np.float32).reshape(1,n,3)/np.float32(10)),
           'atomics':T(np.tile(np.array([0,1],np.float32),(1,n,1))),
           'charges':T(np.tile(np.array([1,0],np.float32),(1,n,1))),
           'mask':T(np.ones((1,n),np.float32))}
    b=np.zeros((1,n,n,5),np.float32);b[...,0]=1
    for i in range(1,n):b[0,i,i-1]=b[0,i-1,i]=np.array([0,1,0,0,0],np.float32)
    prior['bonds']=T(b)
    pocket={'coords':T(np.ones((1,4,3),np.float32)),'atom_names':T(np.ones((1,4),np.int64)),
      'charges':T(np.ones((1,4,2),np.float32)),'bonds':T(np.zeros((1,4,4,5),np.float32)),
      'res_names':T(np.zeros((1,4),np.int64)),'mask':T(np.ones((1,4),np.float32)),
      'complex':[types.SimpleNamespace(com=T(np.array([2.,3.,4.],np.float32)))]}
    times=[T(np.zeros(1,np.float32)) for _ in range(3)]
    inputs={'prior':snap(prior),'pocket_data':snap(pocket),'pocket_com':[pocket['complex'][0].com.clone()],
      'rng':rng_state(),'times':snap(times),'coord_scale':model.coord_scale,'solver':'euler','strategy':'linear','final_inpaint':False}
    calls=[];encoding={}
    def pre(_mod,args,kwargs):
        if not encoding:encoding.update({k:snap(kwargs[k]) for k in ['pocket_equis','pocket_invs']})
        calls.append({'call_index':len(calls),'state_before_forward':snap(args[0]),'times':snap(args[2]),
            'times_scalar':[float(x) for x in args[2]],'self_condition':snap(kwargs['cond_batch']),'rng_before_forward':rng_state()})
    def post(_mod,args,kwargs,out):calls[-1]['network_output_before_postprocessing']=snap(out)
    hp=model.register_forward_pre_hook(pre,with_kwargs=True);hq=model.register_forward_hook(post,with_kwargs=True)
    final=model._generate(prior,pocket,steps=5,times=times,save_traj=True)
    hp.remove();hq.remove()
    data={'inputs':inputs,'calls':calls,'encoding':encoding,'final':snap(final),'manifest':{'args':
          {'integration_steps':5,'corrector_iters':0,'save_traj':True,'coord_noise_scale':0.}}}
    return model,data,[v.copy() for v in model.recorded_writes]
