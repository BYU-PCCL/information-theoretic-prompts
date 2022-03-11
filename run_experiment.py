from postprocessor import postprocess
from lmsampler import LMSampler
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def run_experiment(df, model):
    '''
    Given a dataframe and a language model, apply the language model to the dataframe.
    The new dataframe should have all the same old information, plus 'lm_prediction' column.
    '''
    print('Running experiment...')
    lm_predictions = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        lm_predictions.append(model.send_prompt(row['prompt']))
    df['token_logprobs'] = lm_predictions
    # post process
    df = postprocess(df)
    return df

def aggregate_by_template(df):
    '''
    Given a dataframe, aggregate the data by template.
    '''
    cols_to_aggregate = {'mutual_inf': 'mean'}
    # if accuracy, also take average
    if 'accuracy' in df.columns:
        cols_to_aggregate['accuracy'] = 'mean'
    aggregated_df = df.groupby('template_name').agg(cols_to_aggregate)
    return aggregated_df

if __name__ == '__main__':
    '''
Run the experiment.
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
    '''
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default='squad', help='the dataset to use. Accepted: imdb, squad. Default: squad')
    args.add_argument('--n', default=64, type=int, help='the number of rows to use from the dataset. Default: 64')
    args.add_argument('--seed', default=0, type=int, help='the seed to use for the dataset. Default: 0')
    args.add_argument('--lm_model', default='gpt2-xl', help='the language model to use. Supported models: \
        - GPT-3: \'gpt3-ada\', \'gpt3-babbage\', \'gpt3-curie\', \'gpt3-davinci\', \'ada\', \'babbage\', \'curie\', \'davinci\' \
        - GPT-2: \'gpt2\', \'gpt2-medium\', \'gpt2-large\', \'gpt2-xl\', \'distilgpt2\' \
        - GPT-J: \'EleutherAI/gpt-j-6B\' \
        - GPT-Neo: \'EleutherAI/gpt-neo-2.7B\', \'EleutherAI/gpt-neo-1.3B\', \'EleutherAI/gpt-neo-125M\' \
        - Jurassic: \'j1-jumbo\', \'j1-large\' \
        Default: \'gpt2-xl\'')
    args = args.parse_args()

    dataset = args.dataset
    n = args.n
    seed = args.seed
    lm_model = args.lm_model

    if dataset == 'imdb':
        # this dataset should AT THE MINIMUM include columns: 'prompt', 'template_name'
        from imdb import get_templatized_dataset
        df = get_templatized_dataset(n, seed)
    elif dataset == 'squad':
        from squad import get_templatized_dataset
        df = get_templatized_dataset(n, seed)
    else:
        raise ValueError('Dataset not supported.')
    
    model = LMSampler(lm_model)

    df = run_experiment(df, model)

    # aggregate by template
    agg_df = aggregate_by_template(df)

    # print highest mi template
    print('Highest mutual information template:')
    print(agg_df.sort_values('mutual_inf', ascending=False).head(1))
    print()

    # if accuracy, print highest accuracy template
    if 'accuracy' in df.columns:
        print('Highest accuracy template:')
        print(agg_df.sort_values('accuracy', ascending=False).head(1))
        print()

        accs = agg_df['accuracy'].values
        mis = agg_df['mutual_inf'].values

        # get correlation between accs and mis
        corr = np.corrcoef(accs, mis)[0, 1]
        print(f'Correlation between accuracy and mutual information: {corr}')


        # plot accuracy vs mutual information
        plt.scatter(mis, accs)
        plt.xlabel('Mutual Information')
        plt.ylabel('Accuracy')
        plt.title(f'{lm_model} on {dataset}')

        # line of best fit
        x = np.linspace(mis.min(), mis.max(), 100)
        m, b = np.polyfit(mis, accs, 1)
        plt.plot(x, m*x + b, '-')

        # save to accuracy_vs_mutual_information.pdf
        plt.savefig('accuracy_vs_mutual_information.pdf')
    
        # calculate average acc, best acc, and use that to get the mi_template % from avg to best
        average_acc = agg_df['accuracy'].mean()
        best_acc = agg_df['accuracy'].max()
        mi_template_acc = agg_df.sort_values('mutual_inf', ascending=False).head(1)['accuracy'].values[0]
        mi_template_acc_percent = (mi_template_acc - average_acc) / (best_acc - average_acc)
        print(f'Improvement from average accuracy to best accuracy: {average_acc}')
