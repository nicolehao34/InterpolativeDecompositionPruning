import argparse
import os
import time

from opt import *

import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
# %matplotlib inline
from gptq import *
from modelutils import *
from quant import *
from datautils import *

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, required=True, choices=[
    "opt-125M",
    "opt-1.3B",
])

parser.add_argument("--seed", type=int, required=False, default=0)
parser.add_argument("--num_samples", type=int, required=False, default=128)

pargs = parser.parse_args()

print("Parsed Arguments")
print(pargs)

class arg:
    def __init__(self):
        self.load=""
        self.model=f"facebook/{pargs.model_name}"
        self.seed=pargs.seed
        self.dataset='c4'
        self.nsamples=pargs.num_samples
        self.groupsize=1024
        self.benchmark=0
        self.check=True
        self.new_eval=True
        self.faster_kernel=True
        self.wbits=4
        self.nearest=False
        self.sym=True
        self.save_out=False
        self.preffix_dir = "Ps"
        self.trits=False
        self.percdamp=.01
        self.act_order=False
args=arg()

if args.load:
    model = load_quant3(args.model, args.load)
else:
    print("model")
    model = get_opt(args.model)
    model.preffix_dir = args.preffix_dir
    model.eval()
    
dataloader, testloader = get_loaders(
    args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
)
print(len(dataloader))

if args.wbits < 16 and not args.nearest:
    tick = time.time()
    model.model.decoder.save_out=True
    quantizers = opt_sequential(model, dataloader, DEV, args)
    model.model.decoder.save_out=False
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
    
datasets = ['wikitext2', 'ptb', 'c4'] 

if args.new_eval:
    datasets = ['c4-new']
for dataset in datasets: 
    dataloader, testloader = get_loaders(
        dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print(dataset)
    opt_eval(model, testloader, DEV, args)

if args.save_out:
    opt_pack3(model, quantizers)
    torch.save(model.state_dict(), args.save) 