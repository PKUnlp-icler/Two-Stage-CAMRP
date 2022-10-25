SURFACE_PATH=$1
NORM_PATH=$2
EXTRA_PATH=$3
WORD_PATH=$4 # Sentence path
OUTPUT_PREFIX=$5

RELATION_CLS_PATH=$6
RELATION_ALIGN_CLS_PATH=$7


python wsd_v2.py $SURFACE_PATH $NORM_PATH $WORD_PATH $OUTPUT_PREFIX

python extra_nodes_tag_to_label.py $EXTRA_PATH $OUTPUT_PREFIX.no_r $OUTPUT_PREFIX.no_r.with_extra
python extra_nodes_tag_to_label.py $EXTRA_PATH $OUTPUT_PREFIX.with_r $OUTPUT_PREFIX.with_r.with_extra

python inference_relation_cls.py $RELATION_CLS_PATH $OUTPUT_PREFIX.with_r.with_extra 
python inference_relation_alignment_cls.py $RELATION_ALIGN_CLS_PATH $OUTPUT_PREFIX.with_r.with_extra


python relation_number_to_literal.py $OUTPUT_PREFIX.with_r.with_extra $OUTPUT_PREFIX.with_r.with_extra.ralign $OUTPUT_PREFIX.with_r.with_extra.ralign.literal
bash matrix_to_tuples.sh $OUTPUT_PREFIX.with_r.with_extra $OUTPUT_PREFIX.with_r.with_extra.relation $OUTPUT_PREFIX.no_r.with_extra 
python combine_ralign_with_no_ralign.py $OUTPUT_PREFIX.with_r.with_extra.ralign.literal $OUTPUT_PREFIX.with_r.with_extra.relation.literal.sync_with_no_r $OUTPUT_PREFIX.with_r.with_extra 
python transform_nodes_relations_to_camr_tuples.py $OUTPUT_PREFIX.with_r.with_extra.relation.literal.sync_with_no_r.with_func_words $OUTPUT_PREFIX.no_r.with_extra 
