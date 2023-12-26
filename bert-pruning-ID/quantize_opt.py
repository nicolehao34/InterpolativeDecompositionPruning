from opt import *
import time
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gptq import *
from modelutils import *
from quant import *
from datautils import *
import scipy
from scipy.sparse import csr_matrix
import scipy.linalg
import pickle

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



#=========================================================================

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

#================================================================

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

#================================================================

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

#================================================================
# file=open("Rs/errs.p", 'rb')
# errs=pickle.load(file)

# errs=errs/errs[:,0][:, None]
# print(errs)


for mu in [pargs.mu]:

    print("model")
    model = get_opt(args.model)
    model.eval()



    neuronWeight=0.5714285714285714 # I set this manually, it just refers to the number of flops in fully connected v.s. heads.  
    headsWeight=1-0.5714285714285714
    plt.figure()
    for e in errs:
        plt.semilogy(e)
    plt.show() 



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
    plt.figure()
    plt.semilogy(headsErr.T)
    plt.show()



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
    plt.figure()
    for e in headerrs:
        plt.semilogy(e)
    plt.show() 
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
        savedResults.write("Quantized: {}, {}, {}, {}\n".format(args.seed, mu, dataset, res))

    savedResults.close()

#==================================================================

dataloader, testloader = get_loaders(
    args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
)
print(len(dataloader))

if args.wbits < 16 and not args.nearest:
    tick = time.time()
    model.model.decoder.save=False
    quantizers = opt_sequential(model, dataloader, DEV, args)
    model.model.decoder.save=False
    print(time.time() - tick)

if args.benchmark:
    gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    if len(gpus) > 1:
        opt_multigpu(model, gpus)
    else:
        model = model.to(DEV)
    if args.benchmark:
        input_ids = next(iter(dataloader))[0][:, :args.benchmark]
        benchmark(model, input_ids, check=args.check)
if args.load:
    exit()

datasets = [pargs.task] 
if args.new_eval:
    datasets = ['wikitext2', 'ptb-new', 'c4-new']
for dataset in datasets: 
    dataloader, testloader = get_loaders(
        dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print(dataset)
    opt_eval(model, testloader, DEV, args)

if args.save:
    opt_pack3(model, quantizers)
    torch.save(model.state_dict(), args.save)
    
#==================================================================

for dataset in datasets: 
    dataloader, testloader = get_loaders(
        dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print(dataset)
    opt_eval(model, testloader, DEV, args)

if args.save:
    opt_pack3(model, quantizers)
    torch.save(model.state_dict(), args.save) 