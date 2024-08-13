import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import gradio as gr

nltk.download('wordnet')
nltk.download('omw-1.4')



def create_wordnet_antonym_dataset():
    antonym_pairs = []

    for syn in wn.all_synsets():
        for lemma in syn.lemmas():
            if lemma.antonyms():
                word = lemma.name().replace('_', ' ')
                antonym = lemma.antonyms()[0].name().replace('_', ' ')
                antonym_pairs.append((word, antonym))

    # Create a DataFrame
    df = pd.DataFrame(antonym_pairs, columns=['input_phrase', 'opposite_phrase'])
    return df

# Create the dataset
df = create_wordnet_antonym_dataset()

# Save the dataset to a CSV (optional)
df.to_csv('wordnet_antonyms.csv', index=False)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
print(dataset)