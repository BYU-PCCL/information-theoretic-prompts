from abc import ABCMeta, abstractmethod

class LMSamplerBaseClass(metaclass=ABCMeta):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def send_prompt(self, prompt, n_probs):
        '''
        Sends the given prompt to a LM.
        Arguments:
            prompt (str) a prompt to be sent to LM
            n_probs (int) number of desired output probalities.
        Return:
            dict (str:int) a dictionary of log probabilities of length n_probs
        '''
        pass