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

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    # for _ in range(3):
    #     torch.manual_seed(0)
    #     conv = get_conversation_template("vicuna")
    #     turns = []
    #     steps = []
    #     new_tokens = []
    #     wall_time = []
    #     for j in range(len(question["turns"][:5])):
    #         qs = question["turns"][j]
    #         conv.append_message(conv.roles[0], qs)
    #         conv.append_message(conv.roles[1], None)
    #         conv.stop_str = "</s>"
    #         prompt = conv.get_prompt()
    #         inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    #         input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    #         try:
    #             torch.cuda.synchronize()
    #             start_time = time.time()
    #             output_ids, stats = generate_batched_func(
    #                 input_ids, attention_mask,
    #                 model,
    #                 tokenizer,
    #                 max_new_tokens,
    #                 **kwargs,
    #                 # drafter=drafter,
    #                 # do_sample=do_sample,
    #                 # temperature=temperature,
    #             )

    #             stats.calculate_stats()
    #             new_token = stats.mean_accept_per_sample + stats.steps
    #             step = stats.steps
    #             total_time = stats.wall_time
    #             accept_length_tree = stats.accepted_length.tolist()
    #             accept_lengths_tree.extend(accept_length_tree)
    #             output_ids = output_ids[0][len(input_ids[0]):]
    #             # be consistent with the template's stop_token_ids
    #             if conv.stop_token_ids:
    #                 stop_token_ids_index = [
    #                     i
    #                     for i, id in enumerate(output_ids)
    #                     if id in conv.stop_token_ids
    #                 ]
    #                 if len(stop_token_ids_index) > 0:
    #                     output_ids = output_ids[: stop_token_ids_index[0]]

    #             output = tokenizer.decode(
    #                 output_ids,
    #                 spaces_between_special_tokens=False,
    #             )
    #             if conv.stop_str and output.find(conv.stop_str) > 0:
    #                 output = output[: output.find(conv.stop_str)]
    #             for special_token in tokenizer.special_tokens_map.values():
    #                 if isinstance(special_token, list):
    #                     for special_tok in special_token:
    #                         output = output.replace(special_tok, "")
    #                 else:
    #                     output = output.replace(special_token, "")
    #             output = output.strip()

    #             if conv.name == "xgen" and output.startswith("Assistant:"):
    #                 output = output.replace("Assistant:", "", 1).strip()
                
    #             turns.append(output)
    #             steps.append(int(step))
    #             new_tokens.append(int(new_token))
    #             wall_time.append(total_time)
    #         except RuntimeError as e:
    #             print("ERROR question ID: ", question["question_id"])
    #             output = "ERROR"
    #         conv.messages[-1][-1] = output
    # print('Warmup done')

    accept_lengths_tree = []
    for chunk in tqdm(chunks(questions[:15], batch_size)):
        choices = []
        cur_accept_lengths_tree = []
        turns = []
        steps = []
        new_tokens = []
        wall_time = []
        # for j in range(len(question["turns"][:1])):
        j = 0
        qs = [question["turns"][j] for question in chunk]

        conv = [get_conversation_template("vicuna") for _ in chunk]
        for i, el in enumerate(conv):
            el.append_message(el.roles[0], qs[i])
            el.append_message(el.roles[1], None)
            el.stop_str = "</s>"
        prompts = [el.get_prompt() for el in conv]
        inputs = tokenizer(prompts,
                            add_special_tokens=True,
                            padding="longest", 
                            return_attention_mask=True, 
                            return_tensors="pt",
                            padding_side="left"
                            ).to(model.device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        try:
            # torch.cuda.synchronize()
            output_ids, stats = generate_batched_func(
                    input_ids, attention_mask,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **kwargs,
                    # drafter=drafter,
                    # do_sample=do_sample,
                    # temperature=temperature,
                )
            # torch.cuda.synchronize()

            stats.calculate_stats()
            new_token = stats.mean_accept_per_sample + stats.steps
            step = stats.steps
            total_time = stats.wall_time
            accept_length_tree = stats.accepted_length.tolist()
            accept_lengths_tree.append(accept_length_tree)
            output_ids = output_ids[0][len(input_ids[0]):]

            # if conv.stop_token_ids:
            #     stop_token_ids_index = [
            #         i
            #         for i, id in enumerate(output_ids)
            #         if id in conv.stop_token_ids
            #     ]
            #     if len(stop_token_ids_index) > 0:
            #         output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            # if conv.stop_str and output.find(conv.stop_str) > 0:
            #     output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            # if conv.name == "xgen" and output.startswith("Assistant:"):
            #     output = output.replace("Assistant:", "", 1).strip()
            
            turns.append(output)
            steps.append(step)
            new_tokens.append(new_token)
            wall_time.append(total_time)
            cur_accept_lengths_tree.append(accept_length_tree)
        except RuntimeError as e:
            print("ERROR question ID: ", question["question_id"])
            output = "ERROR"
        for el in conv:
            el.messages[-1][-1] = output
        # torch.cuda.empty_cache()
        choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                        "accept_lengths": cur_accept_lengths_tree})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            # fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens: ", accept_lengths_tree)


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

