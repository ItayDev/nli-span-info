from math import ceil, sqrt
import torch


def collate(batch, padding_token_id, span_drop=0.85, max_spans: int = 1500, max_len: int = None):
    max_text_len = max([len(sample['input_ids']) for sample in batch])
    text_tensor = padd_and_and_collect([sample['input_ids'] for sample in batch], max_text_len, padding_token_id)

    sentence_start_indexes = []
    sentence_end_indexes = []

    for sample in batch:
        premise_len = sample['premise_len'].item()
        hypothesis_len = sample['hypothesis_len'].item()

        start_indexes, end_indexes = calculate_valid_spans_indexes(premise_len, hypothesis_len, span_drop, max_spans)
        sentence_start_indexes.append(torch.LongTensor(start_indexes))
        sentence_end_indexes.append(torch.LongTensor(end_indexes))

    max_span_num = max([len(span) for span in sentence_start_indexes])

    if span_drop != 1:
        sentence_start_indexes = padd_and_and_collect(sentence_start_indexes, max_span_num, max_text_len - 1)
        sentence_end_indexes = padd_and_and_collect(sentence_end_indexes, max_span_num, max_text_len - 1)

    labels = [sample['label'] for sample in batch]

    return [
        text_tensor,
        sentence_start_indexes,
        sentence_end_indexes,
        torch.LongTensor(labels),
    ]


def calculate_valid_spans_indexes(premise_len, hypothesis_len, span_drop, max_spans):
    if 0 > span_drop or span_drop > 1:
        raise ValueError("span_drop valid range is [0,1)")
    elif span_drop == 1:
        return [], []
    start_indexes = []
    end_indexes = []
    hypothesis_start_index = 1 + premise_len + 1
    hypothesis_end_index = hypothesis_start_index + hypothesis_len
    premise_step = calculate_step(premise_len, span_drop, max_spans)
    hypothesis_step = calculate_step(hypothesis_len, span_drop, max_spans)

    for i in range(1, premise_len + 1, premise_step):
        for j in range(i, premise_len + 1, premise_step):
            start_indexes.append(i)
            end_indexes.append(j)

    for i in range(hypothesis_start_index, hypothesis_end_index, hypothesis_step):
        for j in range(i, hypothesis_end_index, hypothesis_step):
            start_indexes.append(i)
            end_indexes.append(j)

    return start_indexes, end_indexes


def padd_and_and_collect(data, max_data_len, padding_token_id):
    data_tensor = torch.full([len(data), max_data_len],
                             fill_value=padding_token_id,
                             dtype=data[0][0].dtype)
    for i, sample in enumerate(data):
        data_tensor[i][:len(sample)] = sample

    return data_tensor


def calculate_step(range_len, span_drop, max_span_size):
    if span_drop == 0:
        return 1

    # Arithmetic progression sum
    number_of_spans = (range_len + 1) * (range_len - 1) / 2
    wanted_number_of_spans = min(ceil(number_of_spans * (1 - span_drop)), max_span_size)

    if wanted_number_of_spans == 0:
        return ceil(range_len)
    else:
        return ceil(range_len / sqrt(wanted_number_of_spans))
