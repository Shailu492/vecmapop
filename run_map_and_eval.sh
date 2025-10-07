#!/bin/bash

# Use GPU 2
export CUDA_VISIBLE_DEVICES=2

source ~/python3-10/.venv/bin/activate

log_file="--log_results_file /home/shay/vecmapop/results/eval_translation_results.txt"
fixed_settings="--eval_translation --acl2018 --verbose --cuda"
no_reweight_whiten="--no_reweight --no_whiten"

en_de="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/de.emb.txt aligned/aligned_en_en-de.emb.txt aligned/aligned_de_en-de.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-de.test.txt"
en_it="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/it.emb.txt aligned/aligned_en_en-it.emb.txt aligned/aligned_it_en-it.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-it.test.txt"
en_es="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/es.emb.txt aligned/aligned_en_en-es.emb.txt aligned/aligned_es_en-es.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-es.test.txt"
en_fi="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/fi.emb.txt aligned/aligned_en_en-fi.emb.txt aligned/aligned_fi_en-fi.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-fi.test.txt"

configs=(
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_de}"
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_it}"
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_es}"
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_fi}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_de}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_it}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_es}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_fi}"
    "${fixed_settings} ${log_file} ${en_de}"
    "${fixed_settings} ${log_file} ${en_it}"
    "${fixed_settings} ${log_file} ${en_es}"
    "${fixed_settings} ${log_file} ${en_fi}"
    "--geomm ${fixed_settings} ${log_file} ${en_de}"
    "--geomm ${fixed_settings} ${log_file} ${en_it}"
    "--geomm ${fixed_settings} ${log_file} ${en_es}"
    "--geomm ${fixed_settings} ${log_file} ${en_fi}"
)

FAILED_CONFIGS=()
SUCCESSFUL_CONFIGS=()

for i in "${!configs[@]}"; do
    config="${configs[$i]}"

    echo "========================================"
    echo "Running configuration $((i+1))/${#configs[@]}: $config"
    echo "========================================"

    if python map_embeddings.py $config; then
        echo "✓ Configuration $((i+1)) completed successfully!"
        SUCCESSFUL_CONFIGS+=("$((i+1))")
    else
        echo "✗ Configuration $((i+1)) failed!"
        FAILED_CONFIGS+=("$((i+1))")
    fi

    echo ""
done

echo "========================================"
echo "Summary:"
echo "Successful: ${#SUCCESSFUL_CONFIGS[@]}/${#configs[@]}"
echo "Failed: ${#FAILED_CONFIGS[@]}/${#configs[@]}"

if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    echo "Failed configurations: ${FAILED_CONFIGS[*]}"
    exit 1
fi

deactivate