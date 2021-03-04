import torch
from torch import nn
from data import NUM_OF_LABELS


class LangInferModel(nn.Module):
    def __init__(self, transformer_base, transformer_config, span_attention_heads):
        super(LangInferModel, self).__init__()
        self.span_attention_heads = span_attention_heads
        hidden_size = transformer_config.hidden_size
        self.transformer = transformer_base
        self.span_info_collect = SICModel(hidden_size)
        self.span_info_extract = SpanInformationExtract(hidden_size, span_attention_heads)
        self.output = nn.Linear(hidden_size * (span_attention_heads + 1), NUM_OF_LABELS)

    def forward(self, input_ids, start_indexs, end_indexs):
        # generate mask
        attention_mask = (input_ids != 1).long()
        # intermediate layer
        res = self.transformer(input_ids, attention_mask=attention_mask)
        classification_hidden_state = res.pooler_output
        hidden_states = res.last_hidden_state
        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(hidden_states, start_indexs, end_indexs)
        H = self.span_info_extract(h_ij)

        # use both Roberta's first token for classification as well as span attention heads
        H = torch.cat([classification_hidden_state, H], dim=1)
        # output layer
        out = self.output(H)
        return nn.functional.softmax(out, dim=1)


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indexs, end_indexs):
        W1_h = self.W_1(hidden_states)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = self.select_indexes(W1_h, start_indexs)
        W2_hj_emb = self.select_indexes(W2_h, end_indexs)
        W3_hi_start_emb = self.select_indexes(W3_h, start_indexs)
        W3_hi_end_emb = self.select_indexes(W3_h, end_indexs)
        W4_hj_start_emb = self.select_indexes(W4_h, start_indexs)
        W4_hj_end_emb = self.select_indexes(W4_h, end_indexs)

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        h_ij = torch.tanh(span)
        return h_ij

    def select_indexes(self, hidden_states, indexes):
        views = [hidden_states.shape[0]] + \
                [1 if i != 1 else -1 for i in range(1, len(hidden_states.shape))]
        expanse = list(hidden_states.shape)
        expanse[0] = -1
        expanse[1] = -1
        indexes = indexes.view(views).expand(expanse)
        return torch.gather(hidden_states, 1, indexes)


class SpanInformationExtract(nn.Module):
    def __init__(self, hidden_size, span_attention_heads):
        super().__init__()
        self.span_weight_extraction = nn.Linear(hidden_size, span_attention_heads)

    def forward(self, h_ij):
        print(h_ij.shape)
        span_attention_weights = self.span_weight_extraction(h_ij)
        span_attention_weights = nn.functional.softmax(span_attention_weights, dim=1)

        # Calculating the weighted avarage for each head in one go - using torch.bmm
        # which allows to do matrix multiplication on batches
        weighted_avarage_spans = torch.bmm(torch.transpose(span_attention_weights, 1, 2), h_ij)
        # spreading all the span attention heads to one long vector
        return weighted_avarage_spans.flatten(start_dim=1, end_dim=2)
