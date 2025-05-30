
export CUDA_VISIBLE_DEVICES=4


################ params start ##############
INPUT_JSON_PATH=/data/yrz/repos/TokenCompose/preprocess_data/input_test.json
OUTPUT_JSON_PATH=/data/yrz/repos/TokenCompose/preprocess_data/output_test.json
OUTPUT_DIR=/data/yrz/repos/TokenCompose/preprocess_data/segmentation_output_dir
################ params end ##############

################ gen sentence noun tags start #############
#python gen_noun_tgt.py \
#    --input_json_path $INPUT_JSON_PATH \
#    --output_json_path $OUTPUT_JSON_PATH \
################ gen sentence noun tags end #############

################ gen mask start #############
python gen_mask.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint model_ckpt/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint model_ckpt/sam_hq_vit_h.pth \
  --use_sam_hq \
  --output_dir $OUTPUT_DIR \
  --output_jsonl $OUTPUT_JSON_PATH \
  --input_metadata $OUTPUT_JSON_PATH \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --device "cuda"
################ gen mask end #############




