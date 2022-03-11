import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def exponentiate(d):
    '''
    Exponentiates a dictionary's probabilities.
    '''
    return {k: np.exp(v) for k, v in d.items()}

def normalize(d):
    '''
    Normalizes a dictionary to sum to one.
    '''
    return {k: v/sum(d.values()) for k, v in d.items()}

def collapse_lower_strip(d):
    '''
    Collapses a dictionary's probabilities after doing lower case and strip, combining where needed.
    '''
    new_d = {}
    for k, v in d.items():
        new_k = k.lower().strip()
        # check if empty
        if new_k:
            # if already present, add to value
            if new_k in new_d:
                new_d[new_k] += v
            # if not present, add to dictionary
            else:
                new_d[new_k] = v
    return new_d

def collapse_token_sets(d, token_sets, matching_strategy='startswith'):
    '''
    Collapses a dictionary's probabilities combining into token sets.
    Args:
        d (dict): Dictionary of probabilities.
        token_sets (dict:str->list, list): Dictionary of token sets, where keys are categories and
            values are lists of tokens. If token_sets is a list, it is assumed that the keys are
            the lists of tokens.
        matching_strategy (str): Strategy for matching tokens. Can be 'startswith' or 'exact'.
    Returns:
        new_d (dict): Dictionary of probabilities after collapsing.
    '''
    # if token_sets is a list, convert to dictionary
    if isinstance(token_sets, list):
        token_sets = {t: [t] for t in token_sets}
    # make sure all items in dictionary are lists
    for k, v in token_sets.items():
        if not isinstance(v, list):
            token_sets[k] = [v]
    # create new dictionary
    new_d = {cat: 1e-10 for cat in token_sets.keys()}
    # iterate over tokens and probs in d
    for token, prob in d.items():
        # iterate over token sets
        for category, tokens in token_sets.items():
            # if token is in the token set, add to new dictionary
            match = False
            for t in tokens:
                if matching_strategy == 'startswith':
                    if t.lower().strip().startswith(token):
                        match = True
                elif matching_strategy == 'exact':
                    if token == t:
                        match = True
            if match:
                new_d[category] += prob
    return new_d

def prob_function(row):
    '''
    Collapses a row of probabilities.
    Args:
        row (pandas.Series): Series of probabilities.
    Returns:
        d (dict: str->float): Dictionary of probabilities after collapsing.
    '''
    d = row['token_logprobs']
    # logprobs to probs
    d = exponentiate(d)
    # lower strip collapse
    d = collapse_lower_strip(d)
    # if 'token_sets' in row, collapse token sets
    if 'token_sets' in row:
        # make sure 'token_sets' isn't None or an empty list
        if row['token_sets']:
            # check if matching strategy exists
            if 'matching_strategy' in row:
                d = collapse_token_sets(d, row['token_sets'], row['matching_strategy'])
            else:
                d = collapse_token_sets(d, row['token_sets'])
    return d

def coverage(d):
    '''
    Returns the sum of the values of d.
    '''
    return sum(d.values())

class Postprocessor:

    def __init__(self, input_df, matching_strategy=None):
        '''
        Instantiates a Postprocessor object.
        matching_strategy (str): Strategy for matching tokens. Can be 'startswith', 'exact', or None.
        '''

        # save dataframe
        self.df = input_df

        self.df['ground_truth'] = self.df['ground_truth'].astype(str)

        # get number of instances where 'token_logprobs' is missing
        num_missing = self.df.loc[self.df.token_logprobs.isnull()].shape[0]
        # print Dropping {} instances with missing token_logprobsonses
        print(f'Dropping {num_missing} instances with missing token_logprobs')
        # drop na where 'token_logprobs' is missing
        self.df = self.df.dropna(subset=['token_logprobs'])

        if matching_strategy:
            if matching_strategy not in ['startswith', 'exact']:
                msg = f"{matching_strategy} is not a valid matching strategy"
                raise RuntimeError(msg)
            self.df['matching_strategy'] = matching_strategy
        self.calculate_probs()
        self.calculate_coverage()
        self.normalize_probs()

        # calculate mutual information
        self.df = self.calculate_mutual_information(self.df)

        # calculate accuracy
        self.df = self.calculate_accuracy(self.df)

        # calculate correct weight
        self.df = self.calculate_correct_weight(self.df)

    def prob_dict_to_arr(self, d):
        '''
        Converts a probability dictionary into an array of probabilities.
        Args:
            d (dict: str->float): dictionary of probabilities for each category/token.
        Returns:
            arr (np.array): array of probabilities.
        '''
        arr = np.array([d[k] for k in d])
        return arr

    def agg_prob_dicts(self, dicts):
        '''
        Given a list of probability dictionaries, aggregate them.
        '''
        n = len(dicts)
        agg_dict = {}
        for d in dicts:
            for k, v in d.items():
                if k not in agg_dict:
                    agg_dict[k] = v / n
                else:
                    agg_dict[k] += v / n
        return agg_dict

    def get_marginal_distribution(self, df, groupby='template_name'):
        '''
        Calculates the marginal distribution over categories.
        '''
        marginal_df = df.groupby(by=groupby)['probs'].agg(self.agg_prob_dicts)
        # series to df
        marginal_df = pd.DataFrame(marginal_df)
        return marginal_df

    def calculate_correct_weight(self, df):
        '''
        Calculates the correct_weight. Adds a column called 'correct_weight' to df.
        df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

        Returns modified df.
        '''
        df = df.copy()

        # Our function for calculating weight on ground truth
        get_correct_weight = lambda row: row['probs'].get(row['ground_truth'], 0)

        # Calculate conditional entropy for each row
        df['correct_weight'] = df.apply(get_correct_weight, axis=1)

        return df

    def entropy(self, arr):
        '''
        Given an array of probabilities, calculate the entropy.
        '''
        return -sum(arr * np.log(arr))

    def calculate_conditional_entropy(self, df):
        '''
        Calculates the conditional entropy, up to a constant. Adds a column called 'conditional_entropy' to df.
        df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

        Returns modified df.
        '''
        df = df.copy()

        entropy_lambda = lambda row: self.entropy(self.prob_dict_to_arr(row['probs']))

        # Calculate entropy for each row
        df['conditional_entropy'] = df.apply(entropy_lambda, axis=1)

        return df

    def calculate_mutual_information(self, df, groupby='template_name'):
        '''
        Calculate the mutual information between the template and the output distribution.
        '''
        # H(Y) - H(Y|X) method
        # first, calculate conditional entropy
        df = self.calculate_conditional_entropy(df)
        # get marginal distributions
        marginal_df = self.get_marginal_distribution(df, groupby)
        # get entropy
        entropy_lambda = lambda row: self.entropy(self.prob_dict_to_arr(row['probs']))
        marginal_df['entropy'] = marginal_df.apply(entropy_lambda, axis=1)
        # function to apply per row
        def mutual_inf(row):
            index = row[groupby]
            mutual_info = marginal_df.loc[index]['entropy'] - row['conditional_entropy']
            return mutual_info

        # apply function to each row
        df['mutual_inf'] = df.apply(mutual_inf, axis=1)

        return df

    def calculate_probs(self):
        '''
        Population the 'probs' column in the dataframe.
        '''
        self.df['probs'] = self.df.apply(prob_function, axis=1)

    def calculate_coverage(self):
        coverage_lambda = lambda row: coverage(row['probs'])
        self.df['coverage'] = self.df.apply(coverage_lambda, axis=1)

    def normalize_probs(self):
        '''
        Normalize the 'probs' column to sum to 1.
        '''
        self.df['probs'] = self.df['probs'].apply(normalize)

    def calculate_accuracy(self, df):
        '''
        Calculates the accuracy of the model. Adds a column called 'accuracy' to df.
        df (pandas.DataFrame): dataframe with columns 'template_name', and 'ground_truth'

        Returns modified df.
        '''
        df = df.copy()

        # if row['ground_truth'] starts with argmax(row['probs']) stripped and lowercase, then it's correct
        def accuracy_lambda(row):
            # guess is argmax of row['probs'] dict
            guess = max(row['probs'], key=row['probs'].get)
            # lower and strip
            guess = guess.lower().strip()
            if row['ground_truth'].lower().strip().startswith(guess):
                return 1
            else:
                return 0
        df['accuracy'] = df.apply(accuracy_lambda, axis=1)

        return df

def postprocess(df, matching_strategy='startswith'):
    '''
    Postprocesses the dataframe.
    '''
    pp = Postprocessor(df, matching_strategy)
    return pp.df