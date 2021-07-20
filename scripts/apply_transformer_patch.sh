#! /bin/bash

filename="/root/anaconda3/envs/pytorch/lib/python3.6/site-packages/transformers/models/bert/modeling_bert.py"
search="math.sqrt(self.attention_head_size)"
replace="8"
if [[ $search != "" && $replace != "" ]]; then
sed -i "s/$search/$replace/" $filename
fi
