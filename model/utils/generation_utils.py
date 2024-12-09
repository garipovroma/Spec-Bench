import torch


def temperature_scaled_softmax(logits, temperature=1.0, dim=0):
    assert not logits.isnan().any()
    if temperature > 0:
        res = torch.softmax(logits / temperature, dim=dim)
    else:
        max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
        res = logits.ge(max_logits).to(dtype=logits.dtype)
    return res


def squeeze_pads(input_ids, attention_mask, pad_token_id):
    bs = input_ids.shape[0]
    squeezed_seqs = [input_ids[i][input_ids[i] != pad_token_id] for i in range(bs)]
    squeezed_masks = [attention_mask[i][input_ids[i] != pad_token_id] for i in range(bs)]

    squeezed_input_ids = torch.nn.utils.rnn.pad_sequence(squeezed_seqs, batch_first=True, padding_side="left", padding_value=pad_token_id)
    squeezed_attention_mask = torch.nn.utils.rnn.pad_sequence(squeezed_masks, batch_first=True, padding_side="left", padding_value=pad_token_id)

    return squeezed_input_ids, squeezed_attention_mask

