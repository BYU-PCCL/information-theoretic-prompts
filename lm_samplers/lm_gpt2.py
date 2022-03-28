from lm_samplers.lmsampler_baseclass import LMSamplerBaseClass
from lm_samplers.lm_utils import get_device_map

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

from pdb import set_trace as breakpoint

class LM_GPT2(LMSamplerBaseClass):
    def __init__(self, model_name):
        super().__init__(model_name)
        '''
        Supported models: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2'
        '''
        # initialize model with model_name
        print(f'Loading {model_name}...')
        # TODO - add GPU support
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # get the number of attention layers
        n_blocks = self.model.config.n_layer
        if torch.cuda.is_available():
            # get all available GPUs
            gpus = np.arange(torch.cuda.device_count())
            self.device = 'cuda:0'
            if len(gpus) > 1:
                device_map = get_device_map(gpus, n_blocks)
                self.model.parallelize(device_map)
            else:
                self.model = self.model.to(self.device)
            print(f'Loaded model on {len(gpus)} GPUs.')
        else:
            self.device = 'cpu'
            print('Loaded model on cpu.')

    def send_prompt(self, prompt, n_probs):
        # encode prompt and pass to model
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # get max number of tokens for model
        max_len = self.model.config.n_ctx
        # if inputs is longer than max_len, take last tokens
        if inputs.shape[1] > max_len:
            inputs = inputs[:, -max_len:]

        with torch.no_grad():
            output = self.model(inputs)

        # get logits for final word (the prediction) from model output
        logits = output.logits[-1][-1].to('cpu')

        # get 'n_probs' predicted tokens associated with the above logits
        tokens = torch.argsort(logits, descending=True)[:n_probs]
        
        # decode tokens into text
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
        # TODO - better way to do this?
        # Sometimes symbols don't come out great in ascii encoding
        preds = [p.encode('ascii', 'ignore').decode('ascii') for p in preds]

        # calculate real probabilities associated with each prediction
        logits_probs = torch.nn.functional.softmax(logits, dim=0)
        probs = torch.argsort(logits_probs, descending=True)[:n_probs]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for i in range(len(preds)):
            self.pred_dict[preds[i]] = np.log(logits_probs[probs[i]].item())

        return self.pred_dict

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        if temperature > 0:
            tokens = self.model.generate(input_ids=inputs, max_new_tokens=n_tokens, do_sample=True, temperature=temperature).to('cpu')
        else:
            tokens = self.model.generate(input_ids=inputs, max_new_tokens=n_tokens, temperature=temperature).to('cpu')
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
        return preds[0][len(prompt)+1:]

if __name__ == '__main__':
    
    model = LM_GPT2('gpt2')
    text = model.sample_several(prompt="What is the capital of France?\nThe capital of France is")
    print(text)