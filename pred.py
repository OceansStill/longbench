import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import os
from pathlib import Path
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
#import fla  # noqa - 导入 FLA 库以注册模型类
#import forgetting_transformer.model
#import forgetting_transformer.tokenizer
sys.path.append("/home/liyijia/LinearAttaetion/edge/cartesia-pytorch")  # 确保 cartesia-pytorch 路径在搜索路径中
from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
from cartesia_pytorch.Llamba.configuration_llamba import LlambaConfig

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "mamba2-760m", "fot-760m", "hgrn2-760m", "delta_net-760m", "transformer-760m"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # if "chatglm3" in model_name:
    #     prompt = tokenizer.build_chat_input(prompt)
    # elif "chatglm" in model_name:
    #     prompt = tokenizer.build_prompt(prompt)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import get_conversation_template
    #     conv = get_conversation_template("vicuna")
    #     conv.append_message(conv.roles[0], prompt)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    # elif "llama2" in model_name:
    #     prompt = f"[INST]{prompt}[/INST]"
    # elif "xgen" in model_name:
    #     header = (
    #         "A chat between a curious human and an artificial intelligence assistant. "
    #         "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    #     )
    #     prompt = header + f" ### Human: {prompt}\n###"
    # elif "internlm" in model_name:
    #     prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model_path, out_path, lock):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device)
    
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # 截断以适应 max_length
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        # 标准输入处理
        input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_data.input_ids.shape[-1]
        
        # 为 Llamba 模型单独处理
        with torch.cuda.device(device):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if "llamba" in model_name.lower():
                    # 明确提供 max_length 和 input_ids
                    input_ids = input_data.input_ids
                    
                    # 确保输入长度不超过限制
                    max_input_length = 5000  # 更保守的限制
                    if input_ids.shape[1] > max_input_length:
                        input_ids = input_ids[:, -max_input_length:]
                        context_length = input_ids.shape[1]
                        print(f"Input truncated to {max_input_length} tokens")
                    
                    # 确保总长度合理
                    target_length = context_length + max_gen
                    
                    # 使用与 LlambaLMWrapper._model_generate 相似的参数
                    output = model.generate(
                        input_ids=input_ids,
                        max_length=target_length,  # 必须明确提供
                       # num_beams=1,
                     #   do_sample=False,
                        temperature=1.0,
                    )[0]
                else:
                    # 其他模型的默认处理方式
                    output = model.generate(
                        **input_data,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
        
        # 处理输出
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        
        # 保存预测结果
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                try:
                    pred.encode("utf-8")
                    json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                except UnicodeEncodeError:
                    warnings.warn("Unicode error encountered.. ignoring invalid stuff")
                    pred = pred.encode("utf-8", errors="ignore").decode("utf-8")
                    json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
                f.flush()
    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "llamba" in model_name.lower():
        # 直接使用 Llamba 的自定义加载方式
        from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
        from cartesia_pytorch.Llamba.configuration_llamba import LlambaConfig
        from transformers import LlamaTokenizer  # Llamba 通常使用 Llama 的分词器
        
        # 加载分词器 (使用与 Llamba 兼容的分词器)
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B",  # 或者从你的模型路径加载
            add_bos_token=True,
            clean_up_tokenization_spaces=False
        )
        
        # 加载模型
        model = LlambaLMHeadModel.from_pretrained(
            path,
            device=device,
            dtype=torch.bfloat16
        )
        model = model.eval()
    else:
        # 其他模型的默认加载方式
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, add_bos_token=True, clean_up_tokenization_spaces=False)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device)
        model = model.eval()
    
    return model, tokenizer

if __name__ == '__main__':

    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    # world_size = 1
    mp.set_start_method('spawn', force=True)

    # model2path = json.load(open("config/model2path.json", "r"))
    # model2path[args.model] = args.model_path
    # model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    model_path = args.model_path
    max_length = args.max_length
    assert args.model == Path(args.model_path).name, f"Model name '{args.model}' is different from the last component of model path '{args.path}'. You can delete this assertion if you are sure this is correct."
    # define your model
    # max_length = model2maxlen.get(model_name, 15500)
    if args.e:
        # datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            # "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        datasets = ["musique","hotpotqa", "trec", "2wikimqa"]
        # datasets = ["triviaqa"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    # "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    # "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        # English tasks
        # datasets = ["2wikimqa", "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "musique", \
                    # "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    # "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        datasets = ["hotpotqa", "musique""trec", "2wikimqa"]
        # datasets = ["2wikimqa"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Even: {args.e}")
        print("=" * 80)
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"

        sentinel_path = f"{out_path}.done"
        if not Path(sentinel_path).exists():
            if Path(out_path).exists():
                print(f"Removing incomplete prediction {out_path}")
                Path(out_path).unlink()
        else:
            # It is done. Not need to redo it
            print(f"Sentinel file exists: {sentinel_path}. Skipping {dataset}.")
            continue


        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        lock = mp.Lock()
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model_path, out_path, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


        for p in processes:
            if p.exitcode != 0:
                raise ValueError(f"Process {p.pid} failed")

        with Path(sentinel_path).open("w"):
            pass
        print("Complete!")