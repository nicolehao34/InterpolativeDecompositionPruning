from opt import *
import time
import argparse

import torch
import torch.nn as nn
from gptq import *
from modelutils import *
from quant import *
from datautils import *
import scipy.linalg

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, required=True, choices=[
    "opt-125M",
    "opt-1.3B",
])

parser.add_argument("--task", type=str, required=True, choices=[
    "wikitext2", 
    "ptb-new", 
    "c4-new"
])

parser.add_argument("--seed", type=int, required=False, default=0)

parser.add_argument("--mu", type=float, required=False, default=0.8)


pargs = parser.parse_args()

attn_head_size=-1
number_of_heads=-1
num_of_layer=-1
holder_size=-1

if pargs.model_name == "opt-125M":
    attn_head_size=768
    number_of_heads=12
    num_of_layer=12
    holder_size=3072
elif pargs.model_name == "opt-1.3B":
    attn_head_size=2048
    number_of_heads=32
    num_of_layer=24
    holder_size=8192
else:
    print("==========================================")
    print("ERROR: attn_head_size, number_of_heads, num_of_layer variables must be set appropriately depending on the opt model you are using. Please change line 25-27 in prune_opt.py.")
    print("==========================================")


class arg:
    def __init__(self):
        self.load=""
        self.model=f"facebook/{pargs.model_name}"
        self.seed=pargs.seed
        self.dataset='c4'
        self.nsamples=128
        self.groupsize=1024
        self.benchmark=0
        self.check=True
        self.new_eval=False
        self.faster_kernel=False
        self.wbits=4
        self.nearest=False
        self.sym=False
        self.save=False
        self.trits=False
        self.percdamp=.01
        self.act_order=False
        self.preffix_dir="Ps"
args=arg()
if args.load:
    model = load_quant3(args.model, args.load)
else:
    print("model")
    model = get_opt(args.model)
    model.eval()

print(model.model.decoder.layers[0].fc2.weight.shape)  
import pickle
file=open('{}/x0.p'.format(args.preffix_dir), 'rb')
x=pickle.load(file)
print(x.shape)
file.close()

import pickle
file=open('{}/x1.p'.format(args.preffix_dir), 'rb')
x1=pickle.load(file)
print(x1.shape)
file.close()
datapts=128

dataHolder=torch.zeros([datapts, 2048, attn_head_size],dtype=torch.float16)

for i in range(0,datapts):
    file=open('{}/x{}.p'.format(args.preffix_dir, i), 'rb')
    x=pickle.load(file)
    #print(x.shape)
    #print(x.shape)
    file.close()
    dataHolder[i]=x


#data=data.to('cuda')

#=================================================================================

from typing import List, Optional, Tuple, Union
def getAttn(
    layerAttn,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    self=layerAttn
    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
    if attn_weights.dtype == torch.float16:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    #attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    #attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value

#=================================================================================

file=open("{}/attn.p".format(args.preffix_dir), 'rb')
attn=pickle.load(file)
file.close()
attn=attn.to('cuda')
attn=torch.cat(datapts*[attn])

#==================================================================================
# Generate Rs


batchsize=1
model=model.to("cuda")
from scipy.sparse import csr_matrix
file=open("{}/attn.p".format(args.preffix_dir), 'rb')
attn=pickle.load(file)
file.close()
attn=attn.to('cuda')
attn=torch.cat(batchsize*[attn])

batches=int((datapts)/ batchsize)
import scipy
size=12800
dimensionFC=int(size/batches)
dimensionHead=int(size/batches)
dimensionNeuron=int(size/batches)
import time

for batch in range(0, int((datapts)/ batchsize)):
    start=time.time()
    print(batch)
    data=dataHolder[batch:batch+batchsize]
    data=data.to("cuda")
    selectRandHead=np.float32(np.random.normal(size=(dimensionHead, 2048*64)))
    selectRandNeuron=np.float32(np.random.normal(size=(dimensionNeuron, 2048)))
    selectRandFC=np.float32(np.random.normal(size=(dimensionFC, 2048)))
    for i, layer in enumerate(model.model.decoder.layers):

        with torch.no_grad():
            
            residual=data
            holder=layer(data, attn)[0]
            data=layer.self_attn_layer_norm(data)
            bsz, tgt_len, _ = data.size()

            hidden_states, self_attn_weights, present_key_value=getAttn(layer.self_attn,hidden_states=data,attention_mask=attn, )
            Z=np.float32(hidden_states.detach().cpu().numpy())
            
            Z=Z.transpose(0,1,3,2)
            Z=Z.reshape((np.prod(Z.shape[:-1]), Z.shape[-1]))
            random=selectRandHead
            
            saveZ=random@Z
            
            np.save("Rs/seed_{}/ZHead{}Data{}.npy".format(args.seed, i, batch), saveZ)
            

            num_attention_heads=number_of_heads

            shape=int(Z.shape[0]/64)
            Z=Z.reshape((shape,64, num_attention_heads))
            Z=Z.transpose((0,2,1))
            Z=Z.reshape((shape, num_attention_heads*64))

            random=selectRandNeuron
            saveZ=random@Z
            
            np.save("Rs/seed_{}/Zneurons{}Data{}.npy".format(args.seed, i, batch), saveZ)


                
            hidden_states, self_attn_weights, present_key_value = layer.self_attn(
                hidden_states=data,  
                attention_mask=attn,
            )

            hidden_states=hidden_states+residual 
            del residual
            hidden_states_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.fc1(hidden_states)
            hidden_states = layer.activation_fn(hidden_states)
            hidden_states=hidden_states.to('cpu').numpy()

            norms=np.linalg.norm(layer.fc2.weight.detach().cpu().numpy(), axis=0)
            hidden_states=hidden_states*norms[None, :]
            
            random=selectRandFC#np.random.normal(size=(dimension, hidden_states.shape[0]))

            hidden_states=random@hidden_states
        
            np.save("Rs/seed_{}/Z{}Data{}.npy".format(args.seed, i, batch), hidden_states)
            

            del data 
            del hidden_states
            data=holder

    print(time.time()-start)
    
matBatch=100
for layer in range(len(model.model.decoder.layers)):
    holder=np.zeros((size, holder_size))
    for i in range(int((datapts)/ batchsize)):
        x=np.load("Rs/seed_{}/Z{}Data{}.npy".format(args.seed, layer, i))
        
        holder[matBatch*i:matBatch*(i+1), :]=x

    np.save("Rs/seed_{}/Z{}.npy".format(args.seed, layer), holder)
print("saved Z")
for layer in range(len(model.model.decoder.layers)):
    holder=np.zeros((size, attn_head_size))
    for i in range(int((datapts)/ batchsize)):

        x=np.load("Rs/seed_{}/Zneurons{}Data{}.npy".format(args.seed, layer, i))
        
        holder[matBatch*i: matBatch*(i+1), :]=x

    np.save("Rs/seed_{}/Zneuron{}.npy".format(args.seed, layer), holder)
print("savedZneuron")
for layer in range(len(model.model.decoder.layers)):

    holder=np.zeros((size, number_of_heads))
    for i in range(int((datapts)/ batchsize)):
        x=np.load("Rs/seed_{}/ZHead{}Data{}.npy".format(args.seed, layer, i))
        
        holder[matBatch*i: matBatch*(i+1), :]=x

    np.save("Rs/seed_{}/ZHead{}.npy".format(args.seed, layer), holder)
    
# ========================================================================================

DEV='cuda'
import scipy
#print(model.model.decoder.layers)
model=model.to("cuda")
# data=dataHolder[:10]
# data=data.to("cuda")

file=open("{}/attn.p".format(args.preffix_dir), 'rb')
attn=pickle.load(file)
file.close()
attn=attn.to('cuda')
attn=torch.cat(10*[attn])


i=0
errs=[]
headerrs=[]

for num, layer in enumerate(model.model.decoder.layers):
    print(num)
    print("layer")
    with torch.no_grad():

       # hidden_states, self_attn_weights, present_key_value=getAttn(layer.self_attn,hidden_states=data,attention_mask=attn, )
        print("attn states shape")

#         file=open("Rs/ZHead{}.p".format(num), 'rb')
#         Z=pickle.load(file)
#         file.close()
        Z=np.load("Rs/seed_{}/ZHead{}.npy".format(args.seed, num))
        
        print(Z.shape)
        #Z=Z.transpose(0,1,3,2)
        #Z=Z.reshape((np.prod(Z.shape[:-1]), Z.shape[-1]))
        print(Z.shape)
        R, P = scipy.linalg.qr(Z, pivoting=True, mode='r')
        num_attention_heads=number_of_heads
        
        
#         file=open("Rs/Zneurons{}.p".format(num), 'rb')
#         Z=pickle.load(file)
#         file.close()        
        Z=np.load("Rs/seed_{}/Zneuron{}.npy".format(args.seed, num))
        P1=np.repeat(P, 64)*64+np.tile(np.arange(64), number_of_heads)
        print(Z.shape)
        Z=Z[:,P1]
        shape=int(Z.shape[0]/64)
        q,R1 = scipy.linalg.qr(Z, pivoting=False, mode='economic')
        
        layerErrs=[]
        sort=np.argsort(P)
        tile=(np.tile(np.arange(64), number_of_heads))
        repeat=(np.repeat(P, 64))
        indx=64*repeat+tile
        for b in range(0,number_of_heads):
            er=q[:,b*64:]@R1[b*64:, b*64:]
            matmul=np.zeros(Z.shape)
            matmul[:, b*64:]=er
            matmul=matmul[:,indx]
            nextLayer=layer.self_attn.out_proj.weight.detach().cpu().numpy()
            layerErrs.append(np.linalg.norm(matmul@nextLayer.T))      
        headerrs.append(layerErrs)
        saved=[R[:number_of_heads,:],P,R1]
        file=open("Rs/seed_{}/HeadR{}.p".format(args.seed, i), 'wb')
        pickle.dump(saved, file)
        file.close()
        
        
        norms=np.linalg.norm(layer.fc2.weight.detach().cpu().numpy(), axis=0)


#         file=open("Rs/Z{}.p".format(num), 'rb')
#         hidden_states=pickle.load(file)
#         file.close()
        hidden_states=np.load("Rs/seed_{}/Z{}.npy".format(args.seed, num))
        print(hidden_states.shape)
        R, P = scipy.linalg.qr(np.float32(hidden_states), pivoting=True, mode='r')
        del hidden_states
        file=open("Rs/seed_{}/R{}.p".format(args.seed, i), 'wb')
        i+=1
        pickle.dump((R[:R.shape[1]],P,norms), file)
        
        del P
        file.close()
        errs.append(np.diag(np.abs(R)))
        del R
#         data=holder
file=open("Rs/seed_{}/HeadsErr.p".format(args.seed), 'wb')
pickle.dump(np.array(headerrs), file)
file.close()


errs=np.array(errs)
print(errs.shape)
file=open("Rs/seed_{}/errs.p".format(args.seed), 'wb')
pickle.dump(np.array(errs), file)
file.close()

# END Generate Rs


file=open("Rs/seed_{}/errs.p".format(args.seed), 'rb')
errs=pickle.load(file)
print(errs)
x=np.arange(1, num_of_layer+1)
a, b=np.polyfit(np.arange(1, num_of_layer+1), np.log(errs[:,0]), deg=1)
reweight=(np.e**(x*a+b))
print(errs)
print(reweight)
errs=errs/reweight[:, None]
file.close()
file=open("Rs/seed_{}/HeadsErr.p".format(args.seed), 'rb')
headerrs=np.array(pickle.load(file))
a, b=np.polyfit(np.arange(1, num_of_layer+1), np.log(headerrs[:,0]), deg=1)
reweight=(np.e**(x*a+b))

headerrs=headerrs/reweight[:, None]
#headerrs[0:8]*=2
weight=1.125**np.arange(number_of_heads, 0, -1)
print(weight.shape)
headerrs=headerrs*weight
#errs[0:6]*=2
file.close()

#====================================================================================

# file=open("Rs/errs.p", 'rb')
# errs=pickle.load(file)

# errs=errs/errs[:,0][:, None]
# print(errs)


# for mu in [.65, .7,.75, .8, .85, .9, .95]:
mu=pargs.mu;

print("model")
model = get_opt(args.model)
model.eval()

neuronWeight=0.5714285714285714 # I set this manually, it just refers to the number of flops in fully connected v.s. heads.  
headsWeight=1-0.5714285714285714

def neurons(mu, H, errs):   # This just calculates what our neuron budget is for a given number of flops and heads
    H=(H)/number_of_heads   # and figures out the right way to distribute them per layer.  
    N=int((mu-(headsWeight*H))/neuronWeight*len(errs[0]))
    if N<0:
        return np.zeros(num_of_layer).astype(int)
    cutoff=-np.sort((-errs.flatten()))[N*len(model.model.decoder.layers)]
    ks=np.sum((errs>cutoff), axis=1)
    return ks

# layerWeight=np.arange(len(model.model.decoder.layers), 0, -1)+1  # Number of densely connected layers after the one we're pruning?  
# layerWeight=np.sqrt(layerWeight)+1
# # layerWeight[0:8]*=2
# # layerWeight[0:12]*=2
# # errs=errs*(layerWeight)[:, None]

# headsErr=headerrs*(layerWeight)[:, None]

headsErr=headerrs

cutoffs=[]
neuronErrors=[]
begin=int(mu*number_of_heads)
                                    # Calculate how many heads and how many neurons to keep.  
for head in range(begin,number_of_heads+1):
    e=-np.sort(-headsErr.flatten())
    ks=neurons(mu, head, errs)
    if head<number_of_heads:
        cutoff=e[head*len(model.model.decoder.layers)]
        heads=np.sum(headsErr>cutoff, axis=1)
    else:
        cutoff=0
        heads=np.ones(number_of_heads)
    cutoffs.append(cutoff)
    mean=[]
    for i in range(0,num_of_layer):
        indx=ks[i]
        file=open("Rs/seed_{}/R{}.p".format(args.seed, i), 'rb')
        R=pickle.load(file)[0]
        file.close()

        mean.append(np.linalg.norm(R[indx:, indx:]))
    mean=np.mean(mean)
    neuronErrors.append(mean)

headsNum=np.argmin(np.abs(cutoffs-2*np.array(neuronErrors)))+begin
kMain=neurons(mu, headsNum, errs)

kMain=np.mean(kMain).astype(int)
print("num of neurons")
print(kMain)
kFull=kMain
print("num of heads")
print(headsNum)


import time
num_attention_heads=number_of_heads

cutoff=-np.sort(-errs.flatten())[kFull*len(model.model.decoder.layers)]
print(cutoff)
import scipy
from scipy.linalg.lapack import dtrtri
ks=np.sum(errs>cutoff, axis=1)
print(ks)
model=model.cpu()
for i, layer in enumerate(model.model.decoder.layers):
    kFull=int(ks[i])
    start=time.time()
    file=open("Rs/seed_{}/R{}.p".format(args.seed, i), 'rb')
    R, P, norms=pickle.load(file)
    print(time.time()-start)
    print(P)
    file.close()
    print(norms)
    kept=norms[P[:kFull]]
    eliminated=norms[P[kFull:]]
    start=time.time()
    print(R[0:kFull, 0:kFull].shape)
    #inv=scipy.linalg.pinv(R[0:kFull, 0:kFull])
    inv=dtrtri(R[0:kFull, 0:kFull], lower=0)[0]
    print(start-time.time())
    start=time.time()

    T = np.concatenate((
            np.identity(kFull),
            ((inv @ R[0:kFull, kFull:None])*kept[:, None])/eliminated[None, :]
            ), axis=1)
    print(time.time()-start)
    start=time.time()
    T = T[:, np.argsort(P)]   
    weights=layer.fc2.weight.detach().cpu().numpy()
    print(weights.shape)
    weights=weights@T.T
    layer.fc2.weight=nn.Parameter(torch.Tensor(weights).type(torch.float16), requires_grad=False)
    layer.fc1.weight=nn.Parameter(layer.fc1.weight[P[0:kFull],:], requires_grad=True)
    layer.fc1.bias=nn.Parameter(layer.fc1.bias[P[0:kFull]], requires_grad=True)
    print(time.time()-start)

heads=headsNum

e=-np.sort(-headsErr.flatten())

if heads<number_of_heads:
    cutoff=e[heads*len(model.model.decoder.layers)]
    ks=np.sum(headsErr>cutoff, axis=1)
    print("head distribution")
    print(ks)    
    for c,layer in enumerate(model.model.decoder.layers):
        k=int(ks[c])
        file=open("Rs/seed_{}/HeadR{}.p".format(args.seed, c), 'rb')
        R,P, R1=pickle.load(file)
        file.close()
        T = np.concatenate((
            np.identity(k),
            dtrtri(R[0:k, 0:k], lower=0)[0] @ R[0:k, k:None]
            ), axis=1)
        T = T[:, np.argsort(P)]
        big=np.repeat(T, 64, axis=1)
        big=np.repeat(big, 64, axis=0)
        mask=np.tile(np.eye(64), (k,num_attention_heads))
        T2=dtrtri(R1[0:k*64, 0:k*64], lower=0)[0] @ R1[0:k*64, k*64:None]
        indx=P[-1]
        subset=P[k:]
        newT=big*mask
        for i in range(number_of_heads-k):
            indx=subset[i]
            newT[:,indx*64:(indx+1)*64 ]=T2[:,i*64:(i+1)*64]
        self=layer.self_attn
        newW=nn.Parameter(torch.Tensor(self.out_proj.weight.detach().numpy()@newT.T).type(torch.float16), requires_grad=True)
        self.out_proj.weight=newW
        qw=self.q_proj.weight.detach().numpy().reshape((num_attention_heads, 64, attn_head_size))
        qb=self.q_proj.bias.detach().numpy().reshape(num_attention_heads, 64)
        qw=qw[P[0:k],:, : ]
        qb=qb[P[0:k],: ]
        qw=qw.reshape((64*k, attn_head_size))
        qb=qb.reshape(64*k)
        qw=nn.Parameter(torch.Tensor(qw).type(torch.float16), requires_grad=True)
        qb=nn.Parameter(torch.Tensor(qb).type(torch.float16), requires_grad=True)
        self.q_proj.weight=qw
        self.q_proj.bias=qb
        qw=self.k_proj.weight.detach().numpy().reshape((num_attention_heads, 64, attn_head_size))
        qb=self.k_proj.bias.detach().numpy().reshape(num_attention_heads, 64)
        qw=qw[P[0:k],:, : ]
        qb=qb[P[0:k],: ]
        qw=qw.reshape((64*k, attn_head_size))
        qb=qb.reshape(64*k)
        qw=nn.Parameter(torch.Tensor(qw).type(torch.float16), requires_grad=True)
        qb=nn.Parameter(torch.Tensor(qb).type(torch.float16), requires_grad=True)
        self.k_proj.weight=qw
        self.k_proj.bias=qb
        qw=self.v_proj.weight.detach().numpy().reshape((num_attention_heads, 64, attn_head_size))
        qb=self.v_proj.bias.detach().numpy().reshape(num_attention_heads, 64)
        qw=qw[P[0:k],:, : ]
        qb=qb[P[0:k],: ]
        qw=qw.reshape((64*k, attn_head_size))
        qb=qb.reshape(64*k)
        qw=nn.Parameter(torch.Tensor(qw).type(torch.float16), requires_grad=True)
        qb=nn.Parameter(torch.Tensor(qb).type(torch.float16), requires_grad=True)
        self.v_proj.weight=qw
        self.v_proj.bias=qb    
        self.num_heads=k
        self.embed_dim=int(64*k)
savedResults=open("output/opt/{}.txt".format(pargs.model_name), 'a')
model=model.cuda()
model=model.to('cuda')
datasets = [pargs.task] 
if args.new_eval:
    datasets = ['wikitext2', 'ptb-new', 'c4-new']
for dataset in datasets: 
    dataloader, testloader = get_loaders(
        dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
    print(dataset)
    res=(opt_eval(model, testloader, DEV, args))
    savedResults.write("{}, {}, {}, {}\n".format(args.seed, mu, dataset, res))

savedResults.close()

#=========================================================================