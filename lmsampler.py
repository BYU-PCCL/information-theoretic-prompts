from lm_samplers.lmsampler_baseclass import LMSamplerBaseClass
from lm_samplers.lm_gpt3 import LM_GPT3
from lm_samplers.lm_gpt2 import LM_GPT2
from lm_samplers.lm_gptj import LM_GPTJ
from lm_samplers.lm_gptneo import LM_GPTNEO
from lm_samplers.lm_jurassic import LM_JURASSIC

class LMSampler(LMSamplerBaseClass):
    '''
    Class to wrap all other LMSampler classes. This way, we can instantiate just by passing a model name, and it will initialize the corresponding class.
    '''
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(model_name)
        '''
        Supported models:
            - GPT-3: 'gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci'
            - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2'
            - GPT-J: 'EleutherAI/gpt-j-6B'
            - GPT-Neo: 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M'
            - Jurassic: 'j1-jumbo', 'j1-large'
        '''
        if model_name in ['gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci']:
            self.model = LM_GPT3(model_name)
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']:
            self.model = LM_GPT2(model_name)
        elif model_name in ['EleutherAI/gpt-j-6B']:
            self.model = LM_GPTJ(model_name)
        elif model_name in ['EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M']:
            self.model = LM_GPTNEO(model_name)
        elif model_name in ['j1-jumbo', 'j1-large']:
            self.model = LM_JURASSIC(model_name)
        else:
            raise ValueError(f'Model {model_name} not supported.')

    def send_prompt(self, prompt, n_probs=100):
        return self.model.send_prompt(prompt, n_probs)

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        return self.model.sample_several(prompt, temperature, n_tokens)
      
            


if __name__ == '__main__':
    model_name = 'gpt2'
    sampler = LMSampler(model_name)
    print(sampler.sample_several('The capital of France is', temperature=0, n_tokens=10))
