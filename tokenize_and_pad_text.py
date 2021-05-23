import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def tokenize_text(df, max_seq, tokenizer):
    return [
        # tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in df.comment_text.values
        tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in df.review.values
    ]


def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])


def tokenize_and_pad_text(df, max_seq, tokenizer):
    tokenized_text = tokenize_text(df, max_seq, tokenizer)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text)


def targets_to_tensor(df, target_columns):
    return torch.tensor(df[target_columns].values, dtype=torch.float32)

def tokenize_and_pad_text_bert(df, device, model_class, tokenizer_class, pretrained_weights, max_seq=128, batch_size=16, target_columns=['label']):
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights).to(device)
    bert_model.eval()

    data_indices = tokenize_and_pad_text(df, max_seq, tokenizer)
    data_indices = data_indices.to(device)

    with torch.no_grad():
        x_data = bert_model(data_indices[:batch_size])[0]
        for idx in tqdm(range(batch_size, len(data_indices), batch_size)):
            x_data = torch.cat((x_data, bert_model(data_indices[idx:idx+batch_size])[0]), 0)
        
    y_data = targets_to_tensor(df, target_columns)

    return x_data, y_data
