from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
import torch
import os,json
import argparse

def convert_lisft_to_hf_divide(input_model_path,path_output):
    import collections
    state_dict = collections.OrderedDict()
    
    input_model=torch.load(input_model_path+"/model.ckpt")
    state_dict["model.embed_tokens.weight"] = input_model["embedding.word.embedding.weight"]

    exprt_num=8
    layers_num=32
    for i in range(layers_num):
        state_dict["model.layers." + str(i) + ".self_attn.q_proj.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.q_proj.weight"]
        state_dict["model.layers." + str(i) + ".self_attn.k_proj.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.k_proj.weight"]
        state_dict["model.layers." + str(i) + ".self_attn.v_proj.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.v_proj.weight"]
        state_dict["model.layers." + str(i) + ".self_attn.o_proj.weight"] = \
            input_model["encoder.transformer." + str(i) + ".self_attn.o_proj.weight"]
        state_dict["model.layers." + str(i) + ".input_layernorm.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"]
        state_dict["model.layers." + str(i) + ".block_sparse_moe.gate.weight"]=input_model["encoder.transformer." + str(i) + ".block_sparse_moe.gate.weight"]
        for e in range(exprt_num):
            state_dict["model.layers." + str(i) + ".block_sparse_moe.experts."+str(e)+".w1.weight"] = \
                    input_model["encoder.transformer." + str(i) + ".block_sparse_moe.experts."+str(e)+".w1.weight"]
            state_dict["model.layers." + str(i) + ".block_sparse_moe.experts."+str(e)+".w2.weight"] = \
                    input_model["encoder.transformer." + str(i) + ".block_sparse_moe.experts."+str(e)+".w2.weight"]
            state_dict["model.layers." + str(i) + ".block_sparse_moe.experts."+str(e)+".w3.weight"] = \
                    input_model["encoder.transformer." + str(i) + ".block_sparse_moe.experts."+str(e)+".w3.weight"]
        state_dict["model.layers." + str(i) + ".post_attention_layernorm.weight"] = \
            input_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"]

    state_dict["model.norm.weight"] = input_model["encoder.layer_norm.weight"]
    state_dict["lm_head.weight"] = input_model["target.lm.output_layer.weight"]

    max_shard_size="10GB"
    max_shard_size = int(max_shard_size) if max_shard_size.isdigit() else max_shard_size
    shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size)
    save_path=path_output
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(save_path, shard_file))
    save_index_file = os.path.join(save_path, WEIGHTS_INDEX_NAME)
    with open(save_index_file, "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    if os.path.exists(input_model_path+"/config.json"):
        os.system("cp "+input_model_path+"/config.json "+save_path)
    if os.path.exists(input_model_path+"/generation_config.json"):
        os.system("cp "+input_model_path+"/generation_config.json "+save_path)
    if os.path.exists(input_model_path+"/tokenizer.json"):
        os.system("cp "+input_model_path+"/tokenizer.json "+save_path)
    if os.path.exists(input_model_path+"/tokenizer_config.json"):
        os.system("cp "+input_model_path+"/tokenizer_config.json  "+save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default=None,required=True,
                        help="Path of the safe model path.")
    
    parser.add_argument("--output_model_path", type=str, default=None,required=True,
                        help="Path of the output bin model path.")

    args = parser.parse_args()

    convert_lisft_to_hf_divide(args.input_model_path,args.output_model_path)

#python convert_fsdp_model_to_hf_model.py --input_model_path /mnt/pfs-mc0p4k/nlu/team/lizr/zhangpei/work/03_lisft/moe/upload/lisft/moe_moel/liptm_moe_pro_no_aux_loss/checkpoint-2271/ --output_model_path /mnt/pfs-mc0p4k/nlu/team/lizr/zhangpei/work/03_lisft/moe/upload/lisft/moe_moel/liptm_moe_pro_no_aux_loss/checkpoint-2271/tmp_out

