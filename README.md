# NetTAG: A Multimodal RTL-and-Layout-Aligned Netlist Foundation Model via Text-Attributed Graph

## 0. data_collect
 * data_js_*: design list for each dataset (JSON file)
 * data_pt/data_pt_pos/data_gnnre: netlists, layouts, and corresponding PT reports

## 1. preprocess (cone generation + processing for each modality)
 * Netlist (text-attributed graph)
     - net2graph_ori(pos):
         * Step1: netlist-to-graph + register cone extraction
         Input: data_collect/data_pt
         Output: ./saved_analyzer for graph + ./saved_graph_split for cone subgraph and expression-annotated node_dict
         ```
         python3 run_parallel.py
         ``` 
         * Step2: Save expression & aumgmentation for ExprLLM pretraining --> 2.1 ExprLLM pretraining
         Input: ./saved_graph_split
         Output: ./saved_expr
         ```
         cd expr_aug
         python3 aug.py
         ```
         * Step3: Use pre-trained ExprLLM to get TAG
         Input:  ./saved_graph_split
         Output: ./save_node_dict_tag (updated node_dict + feature vector)
         ```
         cd model/exprllm/experiments
         python3 run_update_node_dict.py ../infer_configs/expr2vec/Sheard-Llama.json
         ```
         * Step4: Create TAG dataset for TAGFormer
         Input: ./save_node_dict_tag
         Output: dataset/tag/ori(pos)
         ```
         python3 subgraph2dataset_tag.py
         ```
 * RTL (text)
     - rtl2embed
       Input: RTL cone designs
       Output: RTL text embeddings
       ```
       cd rtl2embed/scr
       python3 vlg2vec_nv.py
       ```

 * Layout (graph w. physical characteristics)
     - net2graph_layout:
         * Step1: Same as above
         * Step2: Create graph dataset for layout encoder
         Input: ./saved_graph_split
         Output: dataset/graph/layout
         ```
         python3 subgraph2dataset.py
         ```
## 2. model (ExprLLM + TAGFormer)
 * ExprLLM
   - Model code: model/exprllm/llm2vec
   - Pre-train code: model/exprllm/exprllm/run_exprllm_pretrain.py
   - Inference code: model/exprllm/exprllm/run_update_node_dict.py
 * TAGFormer
   - Model code: model/tagformer/models
   - Pre-train code: model/tagformer/pretrain_net_align.py
   - Inference code: model/tagformer/net2vec_taskx.py
   - Fine-tune code: model/tagformer/finetune_taskx.py
    

