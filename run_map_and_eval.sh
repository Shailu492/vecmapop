#!/bin/bash

# Use GPU 2
export CUDA_VISIBLE_DEVICES=2

source ~/python3-10/.venv/bin/activate

log_file="--log_results_file /home/shay/vecmapop/results/eval_translation_results.csv"
fixed_settings="--eval_translation --acl2018 --verbose --cuda"
no_reweight_whiten="--no_reweight --no_whiten"
longer_opt="--max_opt_iter 150 --max_opt_time 1500"


en_de="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/de.emb.txt aligned/aligned_en_en-de.emb.txt aligned/aligned_de_en-de.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-de.test.txt"
en_it="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/it.emb.txt aligned/aligned_en_en-it.emb.txt aligned/aligned_it_en-it.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-it.test.txt"
en_es="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/es.emb.txt aligned/aligned_en_en-es.emb.txt aligned/aligned_es_en-es.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-es.test.txt"
en_fi="/home/shay/vecmap/data/embeddings/en.emb.txt /home/shay/vecmap/data/embeddings/fi.emb.txt aligned/aligned_en_en-fi.emb.txt aligned/aligned_fi_en-fi.emb.txt --validation /home/shay/vecmap/data/dictionaries/en-fi.test.txt"

wiki_emb_dir="/home/shay/py_geomm/muse_data/vectors"
wiki_dic_dir="/home/shay/py_geomm/muse_data/crosslingual/dictionaries"
wiki_out_dir="/home/shay/vecmapop/aligned/wiki"

wiki_en_es="${wiki_emb_dir}/wiki.en.vec ${wiki_emb_dir}/wiki.es.vec ${wiki_out_dir}/aligned_en_en-es.emb.txt ${wiki_out_dir}/aligned_es_en-es.emb.txt --validation ${wiki_dic_dir}/en-es.5000-6500.txt"
wiki_en_fr="${wiki_emb_dir}/wiki.en.vec ${wiki_emb_dir}/wiki.fr.vec ${wiki_out_dir}/aligned_en_en-fr.emb.txt ${wiki_out_dir}/aligned_fr_en-fr.emb.txt --validation ${wiki_dic_dir}/en-fr.5000-6500.txt"
wiki_en_de="${wiki_emb_dir}/wiki.en.vec ${wiki_emb_dir}/wiki.de.vec ${wiki_out_dir}/aligned_en_en-de.emb.txt ${wiki_out_dir}/aligned_de_en-de.emb.txt --validation ${wiki_dic_dir}/en-de.5000-6500.txt"
wiki_en_ru="${wiki_emb_dir}/wiki.en.vec ${wiki_emb_dir}/wiki.ru.vec ${wiki_out_dir}/aligned_en_en-ru.emb.txt ${wiki_out_dir}/aligned_ru_en-ru.emb.txt --validation ${wiki_dic_dir}/en-ru.5000-6500.txt"
wiki_en_zh="${wiki_emb_dir}/wiki.en.vec ${wiki_emb_dir}/wiki.zh.vec ${wiki_out_dir}/aligned_en_en-zh.emb.txt ${wiki_out_dir}/aligned_zh_en-zh.emb.txt --validation ${wiki_dic_dir}/en-zh.5000-6500.txt"

configs=(
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_es}"
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_fr}"
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_de}"
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_ru}"
    "${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_zh}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_es}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_fr}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_de}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_ru}"
    "--geomm ${no_reweight_whiten} ${fixed_settings} ${log_file} ${wiki_en_zh}"
)

configs2=(
    "--geomm ${longer_opt} ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_de}"
    "--geomm ${longer_opt} ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_it}"
    "--geomm ${longer_opt} ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_es}"
    "--geomm ${longer_opt} ${no_reweight_whiten} ${fixed_settings} ${log_file} ${en_fi}"
)

configs1=(
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