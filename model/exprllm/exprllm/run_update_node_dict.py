import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
from dataclasses import dataclass, field
import numpy as np
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import json, pickle, re
import torch
from torch import nn
from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, is_sat, Ite, Xor, Plus, Equals, Times, Real, GE, LT, LE, GT, Minus, EqualsOrIff
from pysmt.typing import BOOL
from multiprocessing import Pool
import signal
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model
sys.path.append('/home/usr/NetTAG/model/exprllm/')
sys.path.append('/home/usr/NetTAG/model/exprllm/llm2vec')
from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss

from tqdm import tqdm



transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="eos_token",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )


@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                # TODO: Add prepare_for_tokenization here similar to supervised training and see if it impacts performance
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class NCETrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])

        loss = self.loss_function(q_reps, d_reps, d_reps_neg)

        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save config file
        # self.model.config.save_pretrained(output_dir)


def expr2text_attr(n_name, n_tpe, expr, node_out):
    if expr:
        expr = str(expr.serialize())
    expr = f"{n_name} = {expr}"
    expr = re.sub(r"(\\(adder|multiplier|subtractor|comparator)_(\d+)|(sub|add|mul|comp))", "", expr)
    expr = re.sub(r'(/|\\|\'|//)*', '', expr)
    expr = re.sub(r'(_)+\'', '', expr)

    # text_attr = f"{n_name} is a {n_tpe} node with the following symbolic expression: {expr}."
    # text_attr = f"[Name]: {n_name}, [Type]: {n_tpe}, [Symbolic Expression]: {expr}."

    text_attr = (n_name, n_tpe, expr)

    return text_attr


def get_physical_attr(node):
    pwr = node.pwr if node.pwr else 0
    area = node.area if node.area else 0
    delay = node.delay if node.delay else 0
    load = node.load if node.load else 0
    tr = node.tr if node.tr else 0
    prob = node.prob if node.prob else 0
    cap = node.cap if node.cap else 0
    res = node.res if node.res else 0

    feat_vec = [pwr, area, delay, load, tr, prob, cap, res]
    return feat_vec

def get_node_type(node_tpe):
    tpe_lst = ['Input', 'Output', \
               'DFF', 'INV', 'BUF', 'XOR', 'AOI', \
               'OAI', 'OR', 'NAND', 'AND', 'MUX',\
               'XNOR', 'HA', 'FA', 'DLL', 'DLH']
    feat_vec = torch.zeros(len(tpe_lst))
    if node_tpe in tpe_lst:
        feat_vec[tpe_lst.index(node_tpe)] = 1
    return feat_vec

def update_one_node(param):
    n, node = param
    node_text_tuple = expr2text_attr(n, node.tpe, node.in_expr, node.out_expr)
    node_feat_vec = get_physical_attr(node)
    node_tpe_vec = get_node_type(node.tpe)
    node.phy_vec = node_feat_vec
    node.tpe_vec = node_tpe_vec
    phys = f"power: {node_feat_vec[0]}, area: {node_feat_vec[1]}, delay: {node_feat_vec[2]}, load: {node_feat_vec[3]}, toggle rate: {node_feat_vec[4]}, probability: {node_feat_vec[5]}, capacitance: {node_feat_vec[6]}, resistance: {node_feat_vec[7]}"
    node_text_attr = f"[Name]: {node_text_tuple[0]}, [Type]: {node_text_tuple[1]}, [Physical Characteristics]: {{{phys}}}, [Symbolic Expression]: {node_text_tuple[2]}."
    q = ["Please analyze this netlist gate, here is the name, type, physical characteristics and symbolic expression of this gate:", node_text_attr]
    # print(q)
    return (n,node_feat_vec, node_tpe_vec), q

class Node_vec:
    def __init__(self, name, phy_vec, tpe_vec, text_vec):
        self.name = name
        self.phy_vec = phy_vec
        self.tpe_vec = tpe_vec
        self.text_vec = text_vec

def expr2vec_one_design(design_name, l2v):
    print('Current design:', design_name)

    save_dir_ori = f"/home/usr/NetTAG/preprocess/net2graph_{cmd}/save_node_dict_tag"
    if os.path.exists(f"{save_dir_ori}/{design_name}_node_data.pkl"):
        with open (f"{save_dir_ori}/{design_name}_node_data.pkl", 'rb') as f:
            node_data = pickle.load(f)
        with open (f"{save_dir_ori}/{design_name}_text_data.pkl", 'rb') as f:
            text_data = pickle.load(f)
    else:
        node_dict_path = f"../../../preprocess/net2graph_{cmd}/saved_graph_split/{design_name}/{design_name}_node_dict.pkl"
        if not os.path.exists(node_dict_path):
            return
        with open(node_dict_path, 'rb') as f:
            node_dict = pickle.load(f)
        print(f"Total Nodes: {len(node_dict)}")

        node_data = []
        text_data = []
        param_lst = []
        for n, node in node_dict.copy().items():
            node.text_attr = None
            node.phy_vec = None
            node.tpe_vec = None
            node.text_vec = None
            if node.tpe == 'Wire':
                continue
            param_lst.append((n, node))

        with Pool(50) as p:
            ret_data = p.map(update_one_node, param_lst)
            p.close()
            p.join()
        for d in ret_data:
            node_data.append(d[0])
            text_data.append(d[1])

        with open (f"{save_dir_ori}/{design_name}_node_data.pkl", 'wb') as f:
            pickle.dump(node_data, f)
        with open (f"{save_dir_ori}/{design_name}_text_data.pkl", 'wb') as f:
            pickle.dump(text_data, f)
    
    print(len(node_data), len(text_data))

    queries_rep = l2v.encode(text_data)
    
    print('---Updating node_dict---')
    node_lst = []
    vec_arr = np.zeros((len(node_data), 8+17+4096))
    for idx, d in enumerate(node_data):
        n = d[0]
        feat_vec = list(d[1])
        tpe_vec = list(d[2])
        text_vec = list(queries_rep[idx])
        vec = np.concatenate([feat_vec, tpe_vec, text_vec])
        node_lst.append(n)
        vec_arr[idx] = vec
    print('---Updating finish---')
    print(vec_arr.shape)
    node_dict_path = f"../../../preprocess/net2graph_{cmd}/saved_node_dict_tag/{design_name}_node.pkl"
    with open(node_dict_path, 'wb') as f:
        pickle.dump(node_lst, f)
    vec_dict_path = f"../../../preprocess/net2graph_{cmd}/saved_node_dict_tag/{design_name}_vec.npy"
    with open(vec_dict_path, 'wb') as f:
        np.save(f, vec_arr)

class TimeoutException(Exception):
    """Custom exception to raise on a timeout"""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}


    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        attention_dropout=custom_args.simcse_dropout,
    )

    with open ("../../../data_collect/data_js/train_list.json", 'r') as f:
        design_list = json.load(f)
    for design in design_list:
        if design == "FPU":
            continue
        if design != "vga_lcd":
            continue
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60*30)
        try:
            
            expr2vec_one_design(design, l2v)
        except TimeoutException:
            print(f"Timeout for {design}")
            continue
        finally:
            signal.alarm(0)
    


if __name__ == "__main__":
    global cmd
    cmd = "ori"
    # cmd = "pos"
    main()
