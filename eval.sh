# Vicuna_PATH=/your_own_path/vicuna-7b-v1.3
# Eagle_PATH=/your_own_path/EAGLE-Vicuna-7B-v1.3
# Medusa_PATH=/your_own_path/medusa-vicuna-7b-v1.3
# Hydra_PATH=/your_own_path/hydra-vicuna-7b-v1.3
# Space_PATH=/your_own_path/vicuna-v1.3-7b-space
Drafter_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
Model_PATH=TheBloke/Llama-2-7B-Chat-GPTQ
datastore_PATH=./model/rest/datastore/datastore_chat_large.idx
MODEL_NAME=Llamas
TEMP=0.0
GPU_DEVICES=7
BATCH_SIZE=1

bench_NAME="spec_bench"
torch_dtype="float16"
 # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 -m evaluation.inference_sps --model-path $Model_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 -m evaluation_batched.inference_sps_batched --model-path $Model_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-${torch_dtype}-temp-${TEMP}-bs-${BATCH_SIZE} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --batch-size $BATCH_SIZE
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra --model-path $Hydra_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-hydra-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_space --model-path $Space_PATH --model-id ${MODEL_NAME}-space-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype

