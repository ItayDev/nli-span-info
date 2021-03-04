from datasets import load_dataset, load_from_disk
import torch
import os


# type - train or test
def prepare_data(tokenizer):
    columns = ['input_ids', 'premise_len', 'hypothesis_len', 'label']
    snli = load_dataset('snli', cache_dir="cache")
    mnli = load_dataset('glue', 'mnli', cache_dir="cache")

    return (reformat_dataset(snli, tokenizer, ['hypothesis', 'premise']),
            reformat_dataset(mnli, tokenizer, ['hypothesis', 'premise', 'idx'], ['test_matched', 'test_mismatched']))


def reformat_dataset(original_dataset, tokenizer, remove_columns, features_to_remove=[]):
    columns = ['input_ids', 'premise_len', 'hypothesis_len', 'label']
    formatted_dataset = original_dataset.filter(lambda row: row['label'] != -1).map(
        lambda row: encode_data(tokenizer, row),
        remove_columns=remove_columns)
    for feature_to_remove in features_to_remove:
        formatted_dataset.pop(feature_to_remove, None)
    formatted_dataset.set_format('torch', columns=columns)

    return formatted_dataset


def encode_data(tokenizer, data):
    sep_token = tokenizer.sep_token_id
    start_token = tokenizer.bos_token_id
    end_token = tokenizer.eos_token_id
    encoded_premise = tokenizer(data['premise'], add_special_tokens=False)['input_ids']
    encoded_hypothesis = tokenizer(data['hypothesis'], add_special_tokens=False)['input_ids']
    input_ids = [start_token] + encoded_premise + [sep_token] + encoded_hypothesis + [end_token]
    premise_len = len(encoded_premise)
    hypothesis_len = len(encoded_hypothesis)
    label = data['label']

    return {
        'input_ids': torch.LongTensor(input_ids),
        'premise_len': torch.LongTensor([premise_len]),
        'hypothesis_len': torch.LongTensor([hypothesis_len]),
        'label': torch.LongTensor([label])
    }
