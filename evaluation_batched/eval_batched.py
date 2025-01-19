"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from model.utils.statistics import ExperimentStats

def chunks_with_warmup(lst, n):
    for i in range(-n, len(lst), n):
        if i >= 0:
            yield lst[i:i + n], False
        else:
            yield lst[0:0 + n], True

def run_eval_batched(
        model,
        tokenizer,
        generate_batched_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        batch_size,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                generate_batched_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                batch_size,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)

@torch.inference_mode
def inference_step():

@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        generate_batched_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        batch_size,
        **kwargs,
):
    device = model.device
    print("Device:", device)
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    exp_stats = ExperimentStats(batch_size)
    for chunk, is_warmup in tqdm(chunks_with_warmup(questions, batch_size)):
        exp_stats.new_batch()
        # for j in range(len(question["turns"][:1])):
        j = 0
        qs = [question["turns"][j] for question in chunk]

        conv = [get_conversation_template("vicuna") for _ in chunk]
        for b, el in enumerate(conv):
            el.append_message(el.roles[0], qs[b])
            el.append_message(el.roles[1], None)
            el.stop_str = "</s>"
        prompts = [el.get_prompt() for el in conv]
        inputs = tokenizer(prompts,
                            add_special_tokens=True,
                            padding="longest", 
                            return_attention_mask=True, 
                            return_tensors="pt",
                            padding_side="left"
                            ).to(device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            output_ids, stats = generate_batched_func(
                    input_ids, attention_mask,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **kwargs,
                )
            torch.cuda.synchronize()

            output_ids = [output_ids[b][len(input_ids[b]):] for b in range(batch_size)]

            for b, cv in enumerate(conv):
                if cv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in cv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids[b] = output_ids[b][: stop_token_ids_index[b]]

            output = [tokenizer.decode(
                output_ids[b],
                spaces_between_special_tokens=False,
            ) for b in range(batch_size)]

            for b, cv in enumerate(conv):
                if cv.stop_str and output[b].find(cv.stop_str) > 0:
                    output[b] = output[b][: output[b].find(cv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output[b] = output[b].replace(special_tok, "")
                    else:
                        output[b] = output[b].replace(special_token, "")
                output[b] = output[b].strip()

                if cv.name == "xgen" and output[b].startswith("Assistant:"):
                    output[b] = output[b].replace("Assistant:", "", 1).strip()
        except RuntimeError as e:
            print("ERROR question IDs: ", [question["question_id"] for question in chunk], " : ", e)
            output = [f"ERROR"] * batch_size
            stats = None
        for b, el in enumerate(conv):
            el.messages[-1][-1] = output[b]
        choices = exp_stats.update_exp_stats(stats, output, index=1)

        # Dump answers
        if not is_warmup:
            for b, question in enumerate(chunk):
                os.makedirs(os.path.dirname(answer_file), exist_ok=True)
                with open(os.path.expanduser(answer_file), "a") as fout:
                    ans_json = {
                        "question_id": question["question_id"],
                        "category": question["category"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_id,
                        "choices": [choices[b]],
                        "tstamp": time.time(),
                    }
                    fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens per batch: ", exp_stats.get_total_accept_mean())


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

