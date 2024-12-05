import torch

def temperature_scaled_softmax(logits, temperature=1.0, dim=0):
    assert not logits.isnan().any()
    if temperature > 0:
        res = torch.softmax(logits / temperature, dim=dim)
    else:
        max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
        res = logits.ge(max_logits).to(dtype=logits.dtype)
    return res

@torch.no_grad()
def sps_generate(
    input_ids, 
    attention_mask,
    model, 
    tokenizer, 
    max_new_tokens,
    tokens_per_forward=4,
    drafter=None, 
    do_sample=False, 
    temperature=0.0
):  
    cum_accept = torch.zeros(len(input_ids)).to(input_ids.device)
    step = 0
    while True:
        step += 1
        input_ids, accept_length_tree, attention_mask, finished = sps_forward(
            input_ids, attention_mask,
            model,
            tokenizer,
            tokens_per_forward,
            drafter=drafter,
            do_sample=do_sample,
            temperature=temperature,
        )
        cum_accept += accept_length_tree
        if finished.all() or (cum_accept + step >= max_new_tokens).any():
            # Squeeze batch!
            return input_ids, step, cum_accept / step

@torch.no_grad()
def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    max_new_tokens,
    do_sample=False,
    temperature=0,
):
    batch_size = candidate_input_ids.shape[0]
    device = candidate_input_ids.device
    
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]

    q = temperature_scaled_softmax(candidate_logits, temperature, dim=-1)
    q_i = torch.gather(q, 2, new_candidate_input_ids[:, :, None]).squeeze(-1)

    p = temperature_scaled_softmax(new_logits, temperature, dim=-1)
    p_i = torch.gather(p, 2, new_candidate_input_ids[:, :, None]).squeeze(-1)

    probability_ratio = torch.clamp(p_i / q_i, min=0, max=1)

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(dim=-1)
    max_new_tok_tensor = torch.ones((batch_size), dtype=int).to(device) * max_new_tokens
    n_matches = torch.minimum(n_matches, max_new_tok_tensor)
    
    p_n_plus_1 = torch.take_along_dim(p, n_matches.resize(batch_size, 1, 1), 1)
    q_n_plus_1 = torch.take_along_dim(q, n_matches.resize(batch_size, 1, 1), 1)
    p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1 * (n_matches < max_new_tokens).reshape(batch_size, 1, 1)), min=0).squeeze(1)
    if do_sample:
        t = torch.multinomial(p_prime, num_samples=1)
    else:
        t = torch.argmax(p_prime, dim=-1)[:, None]
    return n_matches, t
