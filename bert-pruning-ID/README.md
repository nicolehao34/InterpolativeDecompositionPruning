# STAT: Shrinking Transformers After Training

STAT is a novel transformer retraining-free compression method based on interpolative decompositions. (Insert paper link when available)

# Prerequisites

## Install Dependencies

All of our code is tested in Python 3.7.12 and uses a NVidia GPU with >= 16GB of video memory. Install dependencies using the command below:

```bash
pip3 install -r requirements.txt
```

## Prepare checkpoints

We provide the unpruned checkpoints of BERT-base and DistilBERT used in our experiments which were originally provided by Kwon Et. al. They used the pre-trained transformers provided by [HuggingFace Transformers](https://github.com/huggingface/transformers) and fine-tuned them for 6 GLUE tasks (`mnli, qqp, qnli, sst2, stsb, mrpc`) and 2 SQuAD tasks (`squad, squad_v2`) using standard training recipes.

Links to the checkpoints are provided here:

| Model | Link |
| ---------- | ----------|
| BERT-base | [gdrive](https://drive.google.com/drive/folders/1OWHL7Cjhaf2n67PZX4Pt0Be3Gv2VCLo0?usp=sharing) |
| DistilBERT | [gdrive](https://drive.google.com/drive/folders/1ZyGQL5ynoXs0ffGkENNjHq7eijB-B80l?usp=sharing) |

After downloading the models, create the directory `models`. Place the downloaded checkpoints inside `models` and ensure you have the following directory structure.

```
.
├── dataset
├── efficiency
├── evaluate
├── logs
├── models
│   ├── bert-base-uncased
│   │   ├── mnli
│   │   ├── mrpc
│   │   ├── qnli
│   │   ├── qqp
│   │   ├── squad
│   │   ├── squad_v2
│   │   ├── sst2
│   │   └── stsb
│   └── distilbert-base-uncased 
│       └──(omitted subdirectories but should be identical to bert-base-uncased)
├── output
├── prerequisites
├── prune
└── utils
```

Our framework only accepts HuggingFace Transformers PyTorch models. If you choose to use custom checkpoints, please make sure that each checkpoint directory contains both `config.json` and `pytorch_model.bin` files.

## Prepare Model Classes

Included in this repository are two files under the `prerequisites` directory. Please replace the file at

```
.../<your_python_environment>/python3.7/site-packages/transformers/models/bert/modeling_bert.py
```

with the `modeling_bert.py` file included in `prerequisites`. Please replace the file at 

```
.../<your_python_environment>/python3.7/site-packages/transformers/models/opt/modeling_opt.py
```

with the `modeling_opt.py` file included in `prerequisites`.


# Tasks


## Prune BERT models and test accuracy on GLUE/SQuAD benchmarks

- Supported Models: Bert-base/large, DistilBERT
- Supported Tasks:
    - GLUE: MNLI, QQP, QNLI, SST-2, STS-B, MRPC
    - SQuAD V1.1 and V2
    
This is an example for pruning a QQP BERT-base model with 50% MAC (FLOPs) constraint using 0 for the seed and with a sample batch size of 512:

```
python3 prune_bert.py --model_name bert-base-uncased \
                      --task_name qqp \
                      --mu 0.5 \
                      --seed 0 \
                      --sample_batch_size 512
```

The DistilBERT models can be pruned by changing the `model_name` parameter:

```
python3 prune_bert.py --model_name distilbert-base-uncased \
                      --task_name qqp \
                      --mu 0.5 \
                      --seed 0 \
                      --sample_batch_size 512
```

The results of these runs are located at `output/<model_name>/<task_name>.txt`.

## Prune OPT models and test accuracy on WikiText2/PTB-new/C4-new benchmarks

- Supported Models: `opt-125M, opt-1.3B` (Other OPT models will require tweaking of parameters)
- Supported Tasks: `wikitext2, ptb-new, c4-new`.

This is an example for pruning an OPT-125M model fine tuned on wikitext2 with 70% MAC (FLOPs) constraint using 0 for the seed:

```
python3 opt_eval.py --model_name "opt-125M" \
                    --seed 0

python3 prune_opt.py --model_name "opt-125M" \
                     --task wikitext2 \
                     --mu 0.7 \
                     --seed 0
```

This is an example for quantizing an OPT-125M model to 4 bits with the same parameters as above. The number of bits to quantize to can be set in the file itself, but we do not provide a command line parameter for it.

```
python3 quantize_opt.py --model_name "opt-125M" \
                        --task wikitext2 \
                        --mu 0.7 \
                        --seed 0
```

The results of these runs are located at `output/opt/<model_name>.txt`.


# Citation

A lot of the code in this repository is based off previous work done by Kwon Et. al including the example BERT and DistilBERT checkpoints. Their repository and paper can be found here: [github link](https://github.com/WoosukKwon/retraining-free-pruning). The code related to quantization is based off of previous work done by Frantar Et. al. Their repository and paper can be found here: [github link](https://github.com/IST-DASLab/gptq).

```
INSERT BibTEX citation when available
```
