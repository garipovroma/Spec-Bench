"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation_batched.eval_batched import run_eval_batched, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model.sps.decoding import assisted_decoding

from model.sps_batched.decoding_batched import sps_batched_generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--drafter-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128, # CHANGED FROM ORIGINAL
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--model-quantization",
        type=str,
        default="no",
        choices=["no", "4bit"],
        help="Bits and bytes quantization.",
    )
    parser.add_argument(
        "--drafter-quantization",
        type=str,
        default="no",
        choices=["no", "4bit"],
        help="Bits and bytes quantization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size.",
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    if args.model_quantization == "no":
        model_quantization_config = None
    elif args.model_quantization == "4bit":
        model_quantization_config = BitsAndBytesConfig(load_in4bit=True)
    elif args.model_quantization == "8bit":
        model_quantization_config = BitsAndBytesConfig(load_in8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        # quantization_config=model_quantization_config,
        revision="main",
        device_map="auto"
    )

    if args.drafter_quantization == "no":
        drafter_quantization_config = None
    elif args.drafter_quantization == "4bit":
        drafter_quantization_config = BitsAndBytesConfig(load_in4bit=True)
    elif args.drafter_quantization == "8bit":
        drafter_quantization_config = BitsAndBytesConfig(load_in8bit=True)
    drafter = AutoModelForCausalLM.from_pretrained(
        args.drafter_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        # quantization_config=drafter_quantization_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token_id is None:
        assert model.generation_config.pad_token_id != model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id

        # pad_token = "[PAD]"
        # tokenizer.add_special_tokens({'pad_token': pad_token})
        # model.resize_token_embeddings(len(tokenizer))
        # drafter.resize_token_embeddings(len(tokenizer))
        
    model.eval()
    drafter.eval()

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval_batched(
        model=model,
        tokenizer=tokenizer,
        generate_batched_func=sps_batched_generate,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        drafter=drafter,
        temperature=args.temperature,
        do_sample=do_sample,
        batch_size=args.batch_size,
    )

    reorg_answer_file(answer_file)