# information-theoretic-prompts

## Overview

This is a companion repository to the paper "An Information-Theoretic Approach to Prompt Engineering". This includes examples for how to 

## Setup
To clone repo, run:
```bash
git clone git@github.com:tsor13/information-theoretic-prompts.git
```

## Authors
This code is provided by the authors. For any questions, please reach out to Taylor Sorensen at tsor1313@gmail.com.

## Usage

To generate your own experiment for comparing prompt templates with mutual information, run
```python
python3 run_experiment.py --dataset squad --n 64 --seed 0 --lm_model 'gpt2-xl'
```

The following are the available arguments, although we hope that the code was written in such a way so that the code could be extended to other datasets and models easily.

Supported arguments:
```
Accepts argparse arguments:
    --dataset: the dataset to use. Accepted: imdb. Default: squad
    --n: the number of rows to use from the dataset. Default: 64
    --seed: the seed to use for the dataset. Default: 0
    --lm_model: the language model to use. Supported models:
        - GPT-3: 'gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci'
        - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2'
        - GPT-J: 'EleutherAI/gpt-j-6B'
        - GPT-Neo: 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M'
        - Jurassic: 'j1-jumbo', 'j1-large'
        Default: 'gpt2-xl'
```

## Docker
If you would like to run it in a docker container, there is a dockerfile and command arguments provided. From the parent directory,
```bash
cd docker
sh build.sh
sh run.sh
```
