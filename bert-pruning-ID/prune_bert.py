neuronWeight=0.6622759267443489  # I set this manually, it just refers to the number of flops in fully connected v.s. heads.  
headsWeight=1-0.6622759267443489

import argparse
import logging
import os
import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=[
    'distilbert-base-uncased',
    'bert-base-uncased',
])
parser.add_argument("--task_name", type=str, required=True, choices=[
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "squad",
    "squad_v2",
])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mu", type=float, default=0.5)
parser.add_argument("--sample_batch_size", type=int, default=512)

from scipy.optimize import minimize
from dataset.glue import glue_dataset, max_seq_length, avg_seq_length
from dataset.squad import squad_dataset
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from evaluate.nlp import test_accuracy
from utils.schedule import get_pruning_schedule
from torch import nn

from copy import deepcopy
import scipy


def main():
    args = parser.parse_args()
    seed = args.seed
    task = args.task_name
    percent = args.mu*100
    
    print("model: {}".format(args.model_name))
    print("mu: {}".format(args.mu))
    
    directory = "models/{}/{}".format(args.model_name,task)
    run_id = "{}_{}_{}_{}".format(args.model_name,task,percent,seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed) 
    config = AutoConfig.from_pretrained(directory)
    model_generator = AutoModelForSequenceClassification
    model = model_generator.from_pretrained(directory, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        use_auth_token=None,
    )

    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).to('cuda')

    num_attention_heads=12

    def transpose_for_scores( x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_attention_heads, int(x.shape[-1]/num_attention_heads))
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    resultsFile=open("output/{}/{}.txt".format(args.model_name,task), 'a')

    resultsFile.flush()

    def neurons(mu, H, errs):   # This just calculates what our neuron budget is for a given number of flops and heads
        H=(H)/12   # and figures out the right way to distribute them per layer.  
        N=int((mu-(headsWeight*H))/neuronWeight*3072)
        if N<0:
            return np.zeros(12).astype(int)
        cutoff=-np.sort((-errs.flatten()))[N*len(model.bert.encoder.layer)]
        ks=np.sum((errs>cutoff), axis=1)
        return ks

    ######################################################################################
    # This is the bad way I'm saving the embedding output. Would be great to replace with a method to get x and attn
    # (Which I save in x.p  and attn.p and then reload later)
    preffix_dir = "RP/{}".format(run_id)
    model.bert.save_out=True
    model.bert.preffix_dir = preffix_dir
    sample_batch_size = args.sample_batch_size

    training_dataset = glue_dataset(
        task,
        tokenizer,
        training=True,
        max_seq_len=max_seq_length(task),
        pad_to_max=False,)
                
    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), sample_batch_size).tolist(),
    )


    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )   
    model.eval()
    with torch.no_grad():
        model.bert.save_out=True
        for batch in sample_dataloader:
            for k, v in batch.items():
                batch[k] = v.to("cpu", non_blocking=True)

            outputs = model( **batch)
    del batch
    del outputs
    del k
    del v

    model.bert.save_out=False


    ######################################################################################Test original Accuracy##########
    set_seed(seed)
    print("original test acc")
    test_acc = test_accuracy(model.to('cuda'), None, full_neuron_mask.to('cuda'), tokenizer, task)
    print(test_acc)
    model=model.to('cpu')
            
            
    errs=[]
    ######################################################################################  This is where I reload the x and attn mats....
    file=open('{}attn.p'.format(preffix_dir), 'rb')
    attn=pickle.load(file)
    file.close()
    attn=torch.Tensor(attn).cpu()
    file=open('{}x.p'.format(preffix_dir), 'rb')
    x=pickle.load(file)
    file.close()
    x=torch.Tensor(x).cpu()   
    where=np.where(attn.reshape(attn.shape[0], attn.shape[-1])>-1)
    ############################################################################################### SAVE R,P BLOCK    We only need to do this once per model/random seed, and it saves time! (same r and p can compress to different ratios)  
    for i, layer in enumerate(model.bert.encoder.layer):
        with torch.no_grad():
            nextX=layer(x, attn)[0]
            z=layer.attention(x,attn)[0]
            Z=layer.intermediate(z).detach().numpy()
        Z=Z[where[0], where[1], :]  
        Z1=Z
        norms=np.linalg.norm(layer.output.dense.weight.detach().numpy(), axis=0)
        Z1=Z1*norms[None, :]
        R, P = scipy.linalg.qr(Z1, pivoting=True, mode='r')
        errs.append(np.abs(np.diag(R)))
        file=open("RP/{}R{}.p".format(run_id,i), 'wb')
        pickle.dump(R[:3100,:],file)
        file.close()
        file=open("RP/{}P{}.p".format(run_id,i), 'wb')
        pickle.dump(P,file)
        file.close()   
        del x 
        x=nextX

    file=open('RP/{}errs.p'.format(run_id), 'wb')
    errs=np.array(errs)
    pickle.dump(errs, file)
    file.close()

    # End Block

    print("r, p saved...")

    #######################################################################################.  SAVE RP HEADS BLOCKs.  ##############
    def calculate(model):
        errs=[]
        file=open('{}x.p'.format(preffix_dir), 'rb')
        x=pickle.load(file)
        file.close()
        x=torch.Tensor(x).cpu()  
        for c, layer in enumerate(model.bert.encoder.layer): 
            with torch.no_grad():
                nextX=layer(x, attn)[0]
                self=layer.attention.self
                num_attention_heads=self.num_attention_heads
                Z=layer.attention.self(x, attn)[0].detach()
            Z=transpose_for_scores(Z)
            Z=Z.permute(0,2,3,1)
            Z=Z[where[0], where[1], :, :]
            Z=Z.reshape(np.prod(Z.shape[:-1]), Z.shape[-1]).numpy()
            R, P = scipy.linalg.qr(Z, pivoting=True, mode='r')
            Z=Z[:,P]
            shape=int(Z.shape[0]/64)
            Z=Z.reshape((shape,64, num_attention_heads))
            Z=Z.transpose((0,2,1))
            Z=Z.reshape((shape, num_attention_heads*64))
            q,R1 = scipy.linalg.qr(Z, pivoting=False, mode='economic')
        ##### ERROR BLOCK
            layerErrs=[]
            sort=np.argsort(P)
            tile=(np.tile(np.arange(64), 12))
            repeat=(np.repeat(P, 64))
            indx=64*repeat+tile
            for i in range(0,12):
                er=q[:,i*64:]@R1[i*64:, i*64:]
                matmul=np.zeros(Z.shape)
                matmul[:, i*64:]=er
                matmul=matmul[:,indx]
                nextLayer=layer.attention.output.dense.weight.detach().numpy()
                layerErrs.append(np.linalg.norm(matmul@nextLayer.T))      
            errs.append(layerErrs)
            saved=[R[:14,:],P,R1]
            file=open("RP/{}HeadR{}.p".format(run_id,c), 'wb')
            pickle.dump(saved, file)
            file.close()
            x=nextX
        file=open("RP/{}HeadsErr.p".format(run_id), 'wb')
        pickle.dump(np.array(errs), file)
        file.close()


    calculate(model)
    ##############################################################################################.  Prune Block  ##############
    file=open('RP/{}errs.p'.format(run_id), 'rb')
    errs=pickle.load(file)
    file.close()

    file=open("RP/{}HeadsErr.p".format(run_id), 'rb')
    headsErr=pickle.load(file)
    headsErr=np.array(headsErr)
    file.close()

    layerWeight=np.arange(len(model.bert.encoder.layer), 0, -1)+1  # Number of densely connected layers after the one we're pruning?  
    layerWeight=np.sqrt(layerWeight)+1


    errs=errs*(layerWeight)[:, None]
    headsErr=headsErr*layerWeight[:, None]



    ######################################################################################  Finally to the main loop!  
    for mu in [args.mu]:
        
        cutoffs=[]
        neuronErrors=[]
        begin=int(mu*12)
                                            # Calculate how many heads and how many neurons to keep.  
        for head in range(begin,13):
            e=-np.sort(-headsErr.flatten())
            ks=neurons(mu, head, errs)
            if head<12:
                cutoff=e[head*len(model.bert.encoder.layer)-1]
                heads=np.sum(headsErr>=cutoff, axis=1)
            else:
                cutoff=0
                heads=np.ones(len(model.bert.encoder.layer))
            cutoffs.append(cutoff)
            mean=[]
            for i in range(0,len(model.bert.encoder.layer)):
                indx=ks[i]
                file=open("RP/{}R{}.p".format(run_id,i), 'rb')
                R=pickle.load(file)
                file.close()
                mean.append(np.linalg.norm(R[indx:, indx:]))
            mean=np.mean(mean)
            neuronErrors.append(mean)

        headsNum=np.argmin(np.abs(cutoffs-2*np.array(neuronErrors)))+begin
        kMain=neurons(mu, headsNum, errs)

        kMain=np.mean(kMain).astype(int)

        

        np.random.seed(seed)
        torch.manual_seed(seed)  
        for heads in [headsNum]:  #This is just a loop because I used to loop over number of heads (before auto).  Can be just a simple run once now.  
            print("hidden size:")
            print(kMain)
            print("heads:")
            print(heads) 


            cutoff=-np.sort((-errs.flatten()))[kMain*len(model.bert.encoder.layer)]
            ks=np.sum((errs>cutoff), axis=1)
            print(ks)
            del model
            config.intermediate_size=3072
            model = model_generator.from_pretrained(directory, config=config)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

            cutoff=64
            l=0
            svs=[]
            model=model.to('cpu')
            file=open('{}x.p'.format(preffix_dir), 'rb')
            x=pickle.load(file)
            file.close()
            for i,layer in enumerate(model.bert.encoder.layer):
                norms=np.linalg.norm(layer.output.dense.weight.detach().numpy(), axis=0)
                kFull=ks[i]
                file=open("RP/{}R{}.p".format(run_id,i), 'rb')
                R=pickle.load(file)
                file.close()
                file=open("RP/{}P{}.p".format(run_id,i), 'rb')
                P=pickle.load(file)
                file.close()
                layer.intermediate.dense.weight=nn.Parameter(layer.intermediate.dense.weight[P[0:kFull],:], requires_grad=True)
                layer.intermediate.dense.bias=nn.Parameter(layer.intermediate.dense.bias[P[0:kFull]], requires_grad=True)
                kept=norms[P[:kFull]]
                eliminated=norms[P[kFull:]]
                T = np.concatenate((
                    np.identity(kFull),
                    ((np.linalg.pinv(R[0:kFull, 0:kFull]) @ R[0:kFull, kFull:None])*kept[:, None])/eliminated[None, :]
                    ), axis=1)
                T = T[:, np.argsort(P)]
                
                layer.output.dense.weight=nn.Parameter(torch.Tensor(layer.output.dense.weight.detach().numpy()@T.T), requires_grad=True)
            config.intermediate_size=kFull
            
            calculate(model)

        
###########################################################################################.  HEADS BLOCK.   ##########################################    
# (yeah it's ugly but theres just a lot of setting stuff that needs to happen)
            model=model.to('cpu')
            num_attention_heads=12
            e=-np.sort(-headsErr.flatten())
            cutoff=e[heads*len(model.bert.encoder.layer)-1]
            ks=np.sum(headsErr>=cutoff, axis=1)
            print("head distribution")
            print(ks)
            for c,layer in enumerate(model.bert.encoder.layer):
                k=int(ks[c])
                #k=heads
                file=open("RP/{}HeadR{}.p".format(run_id,c), 'rb')
                R,P, R1=pickle.load(file)
                file.close()
                T = np.concatenate((
                    np.identity(k),
                    np.linalg.pinv(R[0:k, 0:k]) @ R[0:k, k:None]
                    ), axis=1)
                T = T[:, np.argsort(P)]
                big=np.repeat(T, 64, axis=1)
                big=np.repeat(big, 64, axis=0)
                mask=np.tile(np.eye(64), (k,num_attention_heads))
                T2=np.linalg.pinv(R1[0:k*64, 0:k*64]) @ R1[0:k*64, k*64:None]
                indx=P[-1]
                subset=P[k:]
                newT=big*mask
                for i in range(12-k):
                   indx=subset[i]
                   newT[:,indx*64:(indx+1)*64 ]=T2[:,i*64:(i+1)*64]
                self=layer.attention.self
                newW=nn.Parameter(torch.Tensor(layer.attention.output.dense.weight.detach().numpy()@newT.T), requires_grad=True)
                layer.attention.output.dense.weight=newW
                qw=self.query.weight.detach().numpy().reshape((num_attention_heads, 64, 768))
                qb=self.query.bias.detach().numpy().reshape(num_attention_heads, 64)
                qw=qw[P[0:k],:, : ]
                qb=qb[P[0:k],: ]
                qw=qw.reshape((64*k, 768))
                qb=qb.reshape(64*k)
                qw=nn.Parameter(torch.Tensor(qw), requires_grad=True)
                qb=nn.Parameter(torch.Tensor(qb), requires_grad=True)
                self.query.weight=qw
                self.query.bias=qb
                qw=self.key.weight.detach().numpy().reshape((num_attention_heads, 64, 768))
                qb=self.key.bias.detach().numpy().reshape(num_attention_heads, 64)
                qw=qw[P[0:k],:, : ]
                qb=qb[P[0:k],: ]
                qw=qw.reshape((64*k, 768))
                qb=qb.reshape(64*k)
                qw=nn.Parameter(torch.Tensor(qw), requires_grad=True)
                qb=nn.Parameter(torch.Tensor(qb), requires_grad=True)
                self.key.weight=qw
                self.key.bias=qb
                qw=self.value.weight.detach().numpy().reshape((num_attention_heads, 64, 768))
                qb=self.value.bias.detach().numpy().reshape(num_attention_heads, 64)
                qw=qw[P[0:k],:, : ]
                qb=qb[P[0:k],: ]
                qw=qw.reshape((64*k, 768))
                qb=qb.reshape(64*k)
                qw=nn.Parameter(torch.Tensor(qw), requires_grad=True)
                qb=nn.Parameter(torch.Tensor(qb), requires_grad=True)
                self.value.weight=qw
                self.value.bias=qb    
                self.num_attention_heads=k
                layer.attention.self.all_head_size=64*k
                
            ######################################################################################   Test model
            model=model.to('cuda')
            full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).to('cuda')
            set_seed(seed)
            test_acc = test_accuracy(model, None, full_neuron_mask, tokenizer, task)
            print(test_acc)
            resultsFile.write("{},{},{},{},{},{}\n".format(seed, sample_batch_size, args.mu, kMain, heads, test_acc))
            resultsFile.flush()

if __name__ == "__main__":
    main()
