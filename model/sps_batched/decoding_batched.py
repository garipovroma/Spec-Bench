import torch
from ..utils.statistics import BatchStats
from ..utils.generation_utils import temperature_scaled_softmax, squeeze_pads


@torch.no_grad()
def sps_batched_generate(
    input_ids, 
    attention_mask,
    model, 
    tokenizer, 
    max_new_tokens,
    drafted_tokens=4,
    drafter=None, 
    do_sample=False, 
    temperature=0.0,
    max_steps=512,
):  
    assert drafter is not None
    bs = input_ids.shape[0]
    stats = BatchStats(batch_size=bs)
    stats.set_timer()
    is_finished = torch.zeros(bs, dtype=torch.bool, device=input_ids.device)

    for i in range(max_steps):
        generated_step = sps_batched_one_forward(
            input_ids, 
            attention_mask,
            model, 
            tokenizer, 
            drafted_tokens,
            drafter,
            do_sample,
            temperature,
            is_finished,
        )

        input_ids = generated_step["sequences"]
        attention_mask = generated_step["attention_mask"]
        is_finished = generated_step["is_finished"]
        stats.add_accept(generated_step["verified"].reshape(-1, 1))

        need_to_squeeze = i % 20
        if need_to_squeeze:
            input_ids, attention_mask = squeeze_pads(input_ids, attention_mask, model.generation_config.pad_token_id)

        if (is_finished.sum() == bs):
            break
        
        if (stats.get_new_tokens() >= max_new_tokens).any():
            break

    stats.stop_timer()

    return input_ids, stats


def sps_batched_one_forward(
    input_ids, 
    attention_mask,
    model,
    tokenizer,
    drafted_tokens,
    drafter,
    do_sample=False,
    temperature=1,
    is_finished=None,
):
    bs = input_ids.shape[0]
    device = input_ids.device
    if is_finished is None:
        is_finished = torch.zeros(bs, dtype=torch.bool, device=device)
    max_matches = (torch.ones(bs, device=device, dtype=torch.long) * drafted_tokens + 1) * ~is_finished

    drafted = generate_with_pad(input_ids, attention_mask, drafter, do_sample, temperature, drafted_tokens, stop_on_eos=False)
    candidate_input_ids = torch.cat([drafted["sequences"], torch.zeros(bs, 1, dtype=torch.long, device=device)], 1)
    candidate_logits = torch.cat([drafted["scores"][:, -drafted_tokens:], torch.zeros(bs, 1, drafted["scores"].shape[2], device=device)], 1)
    verifier_attention_mask=torch.cat([attention_mask, torch.ones((bs, drafted_tokens), device=device)], 1)
    
    verifier_logits = model.forward(drafted["sequences"], attention_mask=verifier_attention_mask, position_ids=make_padded_pos_ids(verifier_attention_mask))["logits"]
    new_logits = verifier_logits[:, -max_matches.max().item():]

    valid_tokens, n_matches = _speculative_sampling(candidate_input_ids, candidate_logits, drafted_tokens, new_logits, max_matches, do_sample, temperature)
        
    new_attention_mask = (torch.arange(n_matches.max().item(), device=device).repeat((bs,1)) < n_matches.reshape(bs, 1)).to(dtype=torch.long)
    new_attention_mask = torch.cat([new_attention_mask, torch.ones(bs, 1, device=device, dtype=torch.long)], -1)

    valid_tokens = torch.where(new_attention_mask == 1, valid_tokens, model.generation_config.pad_token_id)

    verified = (n_matches + 1) * ~is_finished

    is_finished = (is_finished | (valid_tokens == model.generation_config.eos_token_id).any(-1))

    return {"sequences" : torch.cat([input_ids, valid_tokens], -1),
            "attention_mask" : torch.cat([attention_mask, new_attention_mask], -1), 
            "verified" : verified,
            "is_finished" : is_finished}


@torch.no_grad()
def generate_with_pad(input_ids, attention_mask, model, do_sample=False, temperature=1, max_new_tokens=1, stop_on_eos=True):
    bs = input_ids.shape[0]
    logits = None

    for i in range(max_new_tokens):
        logits = model.forward(input_ids, attention_mask=attention_mask, position_ids=make_padded_pos_ids(attention_mask))["logits"]
  
        if do_sample:
            logits = logits / temperature
            probs = logits[:, -1].softmax(-1)
            new_token = torch.multinomial(probs, 1).to(input_ids.device)
        else:
            new_token = logits[:, -1].argmax(-1, keepdim=True).to(input_ids.device)
        
        input_ids = torch.cat([input_ids, new_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones(bs, 1, device=attention_mask.device)], dim=-1)

        if stop_on_eos and ((input_ids == model.generation_config.eos_token_id).any(-1).sum() == bs):
            break

    return {"sequences" : input_ids, 
            "scores" : logits, 
            "attention_mask" : attention_mask}


def make_padded_pos_ids(att_mask):
    att_mask = torch.Tensor(att_mask)
    return ((att_mask.cumsum(-1) - 1) * att_mask).to(dtype=torch.long)


@torch.no_grad()
def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    max_matches,
    do_sample=False,
    temperature=0,
):
    batch_size = candidate_input_ids.shape[0]
    device = candidate_input_ids.device

    new_candidate_input_ids = candidate_input_ids[:, -max_matches.max().item():]

    q = temperature_scaled_softmax(candidate_logits, temperature, dim=-1)
    q_i = torch.gather(q, 2, new_candidate_input_ids[:, :, None]).squeeze(-1)

    p = temperature_scaled_softmax(new_logits, temperature, dim=-1)
    p_i = torch.gather(p, 2, new_candidate_input_ids[:, :, None]).squeeze(-1)
    probability_ratio = torch.clamp(p_i / q_i, min=0, max=1)
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(dim=-1)
    n_matches = torch.minimum(n_matches, max_matches)
    
    p_n_plus_1 = torch.take_along_dim(p, n_matches.resize(batch_size, 1, 1), 1)
    q_n_plus_1 = torch.take_along_dim(q, n_matches.resize(batch_size, 1, 1), 1)
    p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1 * (n_matches < max_matches-1).reshape(batch_size, 1, 1)), min=0).squeeze(1)

    new_token = torch.multinomial(p_prime, num_samples=1)
    
    if (n_matches > 0).any():
        valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches.max()], new_token), dim=-1)
    else:
        valid_tokens = new_token

    return valid_tokens, n_matches
