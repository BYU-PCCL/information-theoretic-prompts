import pandas as pd
import numpy as np
from datasets import load_dataset


def get_imdb_df(n=32, seed=0):
    '''
    Returns a df with columns 'text', 'label', and 'instance_id' of length n.
    '''
    # import test dataset
    dataset = load_dataset('imdb')['test']

    # to pandas dataframe
    df = pd.DataFrame(dataset)
    # make column 'instance_id' from index
    df['instance_id'] = df.index
    # map 1 to 'positive' and 0 to 'negative' for label
    df['label'] = df['label'].map({1: 'positive', 0: 'negative'})
    # change name 'label' to 'ground_truth'
    df = df.rename(columns={'label': 'ground_truth'})
    # sample 10 rows
    df = df.sample(n, random_state=seed)
    return df

def get_templates():
    '''
    Return a dictionary of templates.
    Each entry should be of the form: template_name: (lambda prompt generator, token_set_dict)
    '''
    token_set_dict = {'positive' : ['positive'], 'negative' : ['negative']}
    templates = {
        'review_follow_up_q0': (lambda row: (f"{row['text']}\n\n"
                                "Was the previous review positive or negative? The previous review was"), token_set_dict),

        'review_follow_up_q1': (lambda row: (f"{row['text']}\n\n"
                                "Was the previous review negative or positive? The previous review was"), token_set_dict),

        'review_follow_up_q2': (lambda row: (f"{row['text']}\n\n"
                                "Was the sentiment of previous review positive or negative? The previous review was"), token_set_dict),

        'review_follow_up_q3': (lambda row: (f"{row['text']}\n\n"
                                "Was the sentiment of previous review negative or positive? The previous review was"), token_set_dict),

        'task_review_classify0' : (lambda row: ("After reading the following review, classify it as positive or negative. \n\nReview: "
                                    f"{row['text']} \n\nClassification:"), token_set_dict),

        'task_review_classify1' : (lambda row: ("After reading the following review, classify it as negative or positive. \n\nReview: "
                                    f"{row['text']} \n\nClassification:"), token_set_dict),         

        'task_review_follow_up0' : (lambda row: ("Read the following movie review to determine the review's sentiment.\n\n"
                                    f"{row['text']}\n\nIn general, was the sentiment positive or negative? The sentiment was"), token_set_dict),

        'task_review_follow_up1' : (lambda row: ("Read the following movie review to determine the review's sentiment.\n\n"
                                    f"{row['text']}\n\nIn general, was the sentiment negative or positive? The sentiment was"), token_set_dict),

        'task_review_follow_up2' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:  "
                                    f"{row['text']}\n\nIn general, was the sentiment positive or negative The sentiment was"), token_set_dict),

        'task_review_follow_up3' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:\n\"\"\"\n"
                                    f"{row['text']}\n\"\"\"\nIn general, was the sentiment positive or negative? The sentiment was"), token_set_dict),

        'task_review_follow_up4' : (lambda row: ("Considering this movie review, determine its sentiment.\n\nReview:\n\"\"\"\n"
                                    f"{row['text']}\n\"\"\"\nIn general, what was the sentiment of the review? The sentiment was"), token_set_dict),

        'story_review_conclusion0' : (lambda row: ("Yesterday I went to see a movie. "
                                    f"{row['text']} Between positive and negative, I would say the movie was"), token_set_dict),

        'q_and_a0' : (lambda row: ('''Q: Is the sentiment of the following movie review positive or negative?\n"""\n'''
                                    f"{row['text']}\n\"\"\"\nA: The sentiment of the movie review was"), token_set_dict),

        'q_and_a1' : (lambda row: ('''Q: Is the sentiment of the following movie review negative or positive?\n"""\n'''
                                    f"{row['text']}\n\"\"\"\nA: The sentiment of the movie review was"), token_set_dict),

        'q_and_a2' : (lambda row: ("Q: Is the sentiment of the following movie review positive or negative?\n"
                                    f"{row['text']} \nA (positive or negative):"), token_set_dict),

        'q_and_a3' : (lambda row: ("Q: Is the sentiment of the following movie review negative or positive?\n"
                                    f"{row['text']} \nA (negative or positive):"), token_set_dict),

        'dialogue0' : (lambda row: ("P1: Could you give me a review of the movie you just saw? "
                                    f"\nP2: Sure, {row['text']} "
                                    "\nP1: So, overall, would you give it a positive or negative review? "
                                    "\nP2: I would give it a"), token_set_dict),

        'dialogue1' : (lambda row: ("P1: Could you give me a review of the movie you just saw? "
                                    f"\nP2: Sure, {row['text']} "
                                    "\nP1: So overall was the sentiment of the movie negative or positive? "
                                    "\nP2: I would give it a"), token_set_dict),

        'dialogue2' : (lambda row: ("P1: How was the movie? "
                                    f"\nP2: {row['text']} "
                                    "\nP1: Would you say your review of the movie is positive or negative? "
                                    "\nP2: I would say my review of the movie is"), token_set_dict),

        'dialogue3' : (lambda row: ("P1: How was the movie? "
                                    f"\nP2: {row['text']} "
                                    "\nP1: Would you say your review of the movie is negative or positive? "
                                    "\nP2: I would say my review review of the movie is"), token_set_dict),
        
    }
    return templates

def templatize(df, templates):
    '''
    Given a dataframe and a dictionary of templates, apply the templates to the dataframe.
    The new dataframe should have all the same old information, plus 'template_name', 'token_sets', and 'prompt' columns.
    '''
    dfs = []
    for template_name, (prompt_generator, token_set_dict) in templates.items():
        template_df = df.copy()
        template_df['template_name'] = [template_name] * len(template_df)
        template_df['token_sets'] = [token_set_dict] * len(template_df)
        template_df['prompt'] = template_df.apply(prompt_generator, axis=1)
        dfs.append(template_df)
    # concatenate
    templatized_df = pd.concat(dfs)
    # reset index
    templatized_df = templatized_df.reset_index(drop=True)
    return templatized_df

def get_templatized_dataset(n=16, seed=0):
    from pdb import set_trace as breakpoint
    df = get_imdb_df(n=n, seed=seed)
    templates = get_templates()
    templatized_df = templatize(df, templates)
    return templatized_df


if __name__ == '__main__':
    # load data
    df = get_imdb_df(n=8)
    templates = get_templates()
    # templatize
    templatized_df = templatize(df, templates)

    lm_model = LMSampler('gpt2-xl')

    results_df = run_lm(templatized_df, lm_model)

    from pdb import set_trace as breakpoint
    
    # post process
    df = postprocess(results_df)
