import pandas as pd
import numpy as np
from datasets import load_dataset

def get_squad_df(n=32, seed=0):
    # import test dataset
    dataset = load_dataset('squad')['train']
    from pdb import set_trace as breakpoint

    # to pandas dataframe
    df = pd.DataFrame(dataset)
    # sample n rows
    df = df.sample(n, random_state=seed)
    # make column 'instance_id' from index
    df['instance_id'] = df.index
    # extract 'ground_truth' from 'answers'['text']
    f = lambda row: row['answers']['text'][0]
    df['ground_truth'] = df.apply(f, axis=1)
    return df

def get_templates():
    '''
    Return a dictionary of templates.
    Each entry should be of the form: template_name: (lambda prompt generator, token_set_dict)
    '''
    token_set = None
    SHOTS = [
        """CONTEXT:
        BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.
        QUESTIONS:
        1) What high school GPA for BYU freshmen have on average?
        Answer: "3.71"
        """,

        """CHAPTER QUIZ
        PASSAGE: BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.
        QUESTIONS:
        1) What high school GPA for BYU freshmen have on average?

        ANSWER KEY: 
        1) 3.71
        """, 

        """P1: BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.
        P2: What high school GPA for BYU freshmen have on average?
        P1: 3.71
        """,

        """\"BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.", "What high school GPA for BYU freshmen have on average?" -> "3.71\""""

        """\"BYU students arrive with superb preparation. The entering class has an average high school GPA of 3.71 (on a 4.0 scale) and an average ACT score that ranks in the 89th percentile nationally. The University consistently places in the top 20 for enrollment of National Merit Scholars.", "What high school GPA for BYU freshmen have on average?" -> "3.71\"
        \"In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of precipitation include drizzle, rain, sleed, snow, graupel, and hail... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud. Short, intense periods of rain in scattered locations are called\"showers\".\", \"What causes precipitation to fall?" -> "gravity\""""
    ]
    templates = {

        'instruction_qa0' : (lambda row: (f"TASK: Using words from the CONTEXT, answer the below QUESTIONS.\n\n"
        f"CONTEXT:\n{row['context']}\n\n"
        f"QUESTIONS:\n1) {row['question']}\n"
        f"Answer: \""), token_set),

        'instruction_qa1' : (lambda row: (f"TASK: Answer the questions below using the phrasing from the context.\n\n"
        f"CONTEXT:\n{row['context']}\n\n"
        f"QUESTIONS:\n1) {row['question']}\n"
        f"Answer: \""), token_set),

        'instruction_qa2' : (lambda row: (f"TASK: Answer the questions below using the phrasing from the context.\n\n"
        f"CONTEXT:\n{row['context']}\n\n"
        f"QUESTIONS:"
        f"1) {row['question']}\nAnswer: \""), token_set),

        'instruction_qa3' : (lambda row: (f"TASK: Answer the questions below using the phrasing from the context.\n\n"
        f"{SHOTS[0]}\n\n"
        f"CONTEXT:\n{row['context']}\n\n"
        f"QUESTIONS:\n1) {row['question']}\n"
        f"Answer: \""), token_set),

        'answer_key0' : (lambda row: (f"CHAPTER QUIZ\n\n"
        f"PASSAGE:\n{row['context']}\n\n"
        f"QUESTIONS:\n1) {row['question']}\n\n"
        f"ANSWER KEY:\n1)"), token_set),

        'answer_key1' : (lambda row: (f"ANSWER KEY:\n\n"
        f"QUESTION1:\n\"{row['context']}\" {row['question']}\n"
        f"ANSWER1:"), token_set),

        'answer_key2' : (lambda row: (f"CHAPTER QUIZ\n\n"
        f"PASSAGE:\n{row['context']}\n\n"
        f"QUESTIONS:\n1) {row['question']}\n"
        f"ANSWER KEY:\n1) "), token_set),

        'answer_key3' : (lambda row: (SHOTS[1] + f"\nCHAPTER QUIZ\n\n"
        f"PASSAGE:\n{row['context']}\n\n"
        f"QUESTIONS:\n1) {row['question']}\n\n"
        f"ANSWER KEY:\n1)"), token_set),

        'dialogue0' : (lambda row: (f"P1: {row['context']}\n"
        f"P2: {row['question']}\n"
        f"P1: The answer is \""), token_set), 

        'dialogue1' : (lambda row: (f"P1 tells P2 some information, P2 asks comprehension questions, and P1 answers.\n\n"
        f"P1: {row['context']}\n"
        f"P2: {row['question']}\n"
        f"P1: The answer is \""), token_set), 

        'dialogue2' : (lambda row: (f"P1: {row['context']}\n"
        f"P2: {row['question']}\n"
        f"P1:"), token_set),

        'dialogue3' : (lambda row: (SHOTS[2] + f"\n\nP1: {row['context']}\n"
        f"P2: {row['question']}\n"
        f"P1:"), token_set),

        "old0": (lambda row: ("Context: "f"{row['context']}" "\n\nQ: "f"{row['question']}""\n\nA:"), token_set),

        "old1": (lambda row: (f"{row['context']}" "\n\n"f"{row['question']}\n"
        f"The correct answer is:"), token_set),
    
        "old2": (lambda row: ("I read this in a book today:\n"f"{row['context']}" "\n"f"{row['question']}\nAnswer:"), token_set),
        
        "old3": (lambda row: ("I read this in a book today:\n"f"{row['context']}" "\nFrom that context, did you catch "f"{row['question']}\n"
        f"Yes, the answer is"), token_set),
        
        "old4": (lambda row: ("A friend of mine told me this:\n"f"{row['context']}\n"
        f"My friend then asked: {row['question']}\n"
        f"I answered:"), token_set),

        "openai0_shot": (lambda row: ("Given the following passages and questions, provide a brief, correct answer from the text.\n"
        f"\"{row['context']}\", \"{row['question']}\" -> \""), token_set),

        "openai1_shot": (lambda row: ("Given the following passages and questions, provide a brief, correct answer from the text.\n\n" +
        SHOTS[-1] + "\n" +
        f"\"{row['context']}\", \"{row['question']}\" -> \""), token_set),

        "openai2_shot": (lambda row: ("Given the following passages and questions, provide a brief, correct answer from the text.\n\n" +
        SHOTS[-1] + "\n" +
        f"\"{row['context']}\", \"{row['question']}\" -> \""), token_set),

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
    df = get_squad_df(n=n, seed=seed)
    templates = get_templates()
    templatized_df = templatize(df, templates)
    return templatized_df