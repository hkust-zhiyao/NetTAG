import copy, sys, time, json
from DG import *
import networkx as nx
import re
from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, is_sat, Ite, Xor, Plus, Equals, Times, Real, GE, LT, LE, GT, Minus
from pysmt.typing import BOOL
import matplotlib.pyplot as plt
from collections import deque
from multiprocessing import Pool

# sys.setrecursionlimit(10000)

class AST_analyzer(object):
    def __init__(self, ast):
        self.__ast = ast
        self.graph = Graph()
        self.oper_label = 0
        self.const_label = 0
        self.wire_set = set()

        self.wire_dict = {}
        self.temp_dict = {}

        self.func_set = set()
        self.func_dict = {}        

    def AST2Graph(self, ast):
        self.traverse_AST(ast)
        self.graph.cal_node_width()
        self.graph_wire = copy.deepcopy(self.graph)
        self.eliminate_wires(self.graph)

       
    def traverse_AST(self, ast):
        node_type = ast.get_type()
        self.add_decl_node(ast, node_type)
        self.add_instance(ast, node_type)
        
        for c in ast.children():
            self.traverse_AST(c)
    
  
    def add_decl_node(self, ast, node_type):
        if node_type == 'Decl':
            ll = len(ast.children())
            if ll == 1:
                child = ast.children()[0]
                child_type = child.get_type()
                name = child.name
                width = self.get_width(child)
                self.graph.add_decl_node(name, child_type, width, None, child_type)
                if child_type == 'Wire':
                    self.wire_set.add(name)
            
            elif ll >= 2:
                for child in ast.children():
                    child_type = child.get_type()
                    name = child.name
                    width = self.get_width(child)
                    self.graph.add_decl_node(name, child_type, width, None, child_type)
                    if child_type == 'Wire':
                        self.wire_set.add(name)
            
            else:
                print(ll)
                print(ast)
                print(ast.children())
                print(ast.children()[1].name)
                assert False

    def add_instance(self, ast, node_type):
        if node_type == 'InstanceList':
            assert len(ast.instances) == 1
            inst = ast.instances[0]
            inst_name = inst.name
            inst_module = inst.module
            inst_tpe, strength = self.convert_inst_type(inst_module)

        #     if inst_tpe in ['HA', 'FA']:
        #         self.graph.add_decl_node(inst_name+"CO", inst_module, 1, None, inst_tpe+"CO", ast.lineno)
        #         self.graph.add_decl_node(inst_name+"S", inst_module, 1, None, inst_tpe+"S", ast.lineno)
        #         if not inst_tpe:
        #             return
        #         for port_arg in inst.portlist:
        #             port_name = port_arg.portname
        #             if not port_arg.argname:
        #                 continue
        #             port_node = self.add_new_node(port_arg.argname)
        #             assert port_node in self.graph.node_dict.keys()
        #             direc = self.convert_port_direction(port_name, inst_tpe)
        #             if direc == 'i':
        #                 self.graph.add_edge(port_node, inst_name+"CO")
        #                 self.graph.add_edge(port_node, inst_name+"S")
        #             elif direc == 'o':
        #                 if port_name == 'CO':
        #                     self.graph.add_edge(inst_name+"CO", port_node)
        #                 elif port_name == 'S':
        #                     self.graph.add_edge(inst_name+"S", port_node)
            # else:
            self.graph.add_decl_node(inst_name, inst_module, 1, None, inst_tpe, ast.lineno)
            if not inst_tpe:
                return
            for port_arg in inst.portlist:
                port_name = port_arg.portname
                if not port_arg.argname:
                    continue
                port_node = self.add_new_node(port_arg.argname)
                assert port_node in self.graph.node_dict.keys()
                direc = self.convert_port_direction(port_name, inst_tpe)
                if direc == 'i':
                    self.graph.add_edge(port_node, inst_name, port_name)
                elif direc == 'o':
                    self.graph.add_edge(inst_name, port_node, port_name)
                

    def convert_inst_type(self, inst_tpe):
        if re.search(r'^(S)*DFF(R)*(S)*(\d)*_X(\d+)', inst_tpe):
        # if 'DFF' in inst_tpe:
            ret_tpe = 'DFF'
        elif re.search(r'^INV(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'INV'
        elif re.search(r'^BUF(\d)*_X(\d+)|^CLKBUF(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'BUF'
        elif re.search(r'^XOR(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'XOR'
        elif re.search(r'^AOI(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'AOI'
        elif re.search(r'^AO(\w)*_X(\d+)', inst_tpe):
            ret_tpe = 'AOI'
        elif re.search(r'^OAI(\w)*_X(\d+)', inst_tpe):
            ret_tpe = 'OAI'
        elif re.search(r'^OA(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'OAI'
        elif re.search(r'^OR(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'OR'
        elif re.search(r'^NAND(\w)*_X(\d+)', inst_tpe):
            ret_tpe = 'NAND'
        elif re.search(r'^AND(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'AND'
        elif re.search(r'^MUX(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'MUX'
        elif re.search(r'^MX(\w)*_X(\d+)', inst_tpe):
            ret_tpe = 'MUX'
        elif re.search(r'^NOR(\w)*_X(\d+)', inst_tpe):
            ret_tpe = 'NOR'
        elif re.search(r'^XNOR(\w)*_X(\d+)', inst_tpe):
            ret_tpe = 'XNOR'
        elif re.search(r'^HA(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'HA'
        elif re.search(r'^ADDH(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'HA'
        elif re.search(r'^FA(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'FA'
        elif re.search(r'^ADDF(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'FA'
        elif re.search(r'^DLL(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'DLL'
        elif re.search(r'^TBUF(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'BUF'
        elif re.search(r'^BUFH(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'BUF'
        elif re.search(r'^CLKGATETST(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'BUF'
        elif re.search(r'^TINV(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'INV'
        elif re.search(r'^DLH(\d)*_X(\d+)', inst_tpe):
            ret_tpe = 'DLH'
        elif re.search(r'^([A-Z])+(\d)*_X(\d+)', inst_tpe):
            print(inst_tpe)
            assert False

        # elif inst_tpe in ['IBuf_0']:
        #     ret_tpe = ''
        else:
            print(inst_tpe)
            ret_tpe = ''
            assert False
            input()
            # assert False


        strength = 0
        strength_re = re.findall(r"(\S+)_X(\d+)", inst_tpe)
        if strength_re:
            strength = int(strength_re[0][-1])

        return ret_tpe, strength  

    def convert_port_direction(self, port_name, inst_tpe=None):
            
        if inst_tpe in ['HA', 'FA']:
            if port_name in ['A', 'B', 'CI']:
                return 'i'
            elif port_name in ['S', 'CO']:
                return 'o'
            else:
                print(port_name)
                assert False
        else:
            if port_name in ['D', 'CK', 
                            'A', 'A0', 'A1', 'A2', 'A3', 'A4', 'A1N', 'A0N', 'AN',\
                            'B', 'B0', 'B1', 'B2', 'B3', 'B4', 'B0N', 'BN',\
                            'C', 'C0', 'C1', 'C2', \
                            'DN',
                            'RN', 'SI', 'SE', 'SN', 'S0',\
                            'GN', 'EN', 'I', 'G', 'E']:
                return 'i'
            elif port_name in ['Q', 'QN', 'Z', 'ZN', "Y", "CO", 'S', 'GCK']:
                return 'o'
            else:
                print(port_name)
                assert False
    
    def add_parent_edge(self):
        for name, node in self.graph.node_dict.items():
            if node.father:
                # if self.graph.node_dict[node.father].type == 'Reg':
                    self.graph.add_edge(node.father, name)

    def cal_width(self, ast):
        msb = int(ast.msb.value)
        lsb = int(ast.lsb.value)
        LHS = max(msb, lsb)
        RHS = min(msb, lsb)
        width = LHS - RHS + 1
        return width

    def get_width(self, ast): # -> int
        width = ast.width
        dimens = ast.dimensions
        if width:
            width = self.cal_width(width)
        else:
            width = 1
        if dimens:
            length = dimens.lengths[0]
            length = self.cal_width(length)
        else:
            length = 1
        return width*length



    def get_node_width(self, ast):
        node_type = ast.get_type()
        parent_type = ast.get_parent_type()
        
        if node_type == 'Identifier':
            width = self.graph.node_dict[ast.name].width

        elif node_type == 'Pointer':
            width = 1
        elif node_type == 'Partselect':
            self.add_new_node(ast)
            width = self.graph.node_dict[ast.var.name].width
        elif node_type == 'IntConst':
            width = self.get_width_num(ast.value)
        elif node_type in ['Concat']:
            width = None
        elif parent_type == 'UnaryOperator':
            width = self.get_node_width(ast.right)
        else:
            print(node_type)
            assert False

        return width


            
    def add_new_node(self, ast):
        node_type = ast.get_type()
        parent_type = ast.get_parent_type()
        if node_type == 'Identifier':
            node_name = ast.name
            assert node_name in self.graph.node_dict.keys()
        elif node_type == 'Pointer':
            name = ast.var.name
            ptr = ast.ptr.value
            node_name = name + '_reg_' + ptr + '_'
            if node_name not in self.graph.node_dict.keys():
                self.graph.add_decl_node(node_name, 'Pointer', 1, name)
        elif node_type == 'Partselect':
            name = ast.var.name
            if (ast.msb.get_type() != 'IntConst' or ast.msb.get_type() != 'IntConst'):
                node_name = name
            else:
                msb = ast.msb.value
                lsb = ast.lsb.value
                width = self.cal_width(ast)
                node_name = name + '_ps_' + msb + '_' + lsb
                if node_name not in self.graph.node_dict.keys():
                    self.graph.add_decl_node(node_name, 'Partselect', width, name)
        elif node_type == 'IntConst':
            node_name = 'Const'
            self.graph.add_decl_node(node_name, 'Const', 1, None, 'Const')
        else:
            print(node_type)
            assert False

        

        return node_name
    
   

    def get_width_num(self, num):
        is_string = re.findall(r"[a-zA-Z]+\'*[a-z]* |'?'*", num)
        if num in ['0', '1']:
            width = 1    
        elif '\'' in num:
            width = re.findall(r"(\d+)'(\w+)", num)
            width = int(width[0][0])
        elif is_string:
            width = len(num)
        else:
            print('ERROR: New Situation!')
            print(num)
            width = 0
            print(is_string)
            assert False
        
        return width

    
    def eliminate_wires(self, g:Graph):
        print('----- Eliminating Wires in Graph -----')
        for name, node in self.graph.node_dict.items():
            if node.father in self.wire_set:
                # print(name)
                # input()
                self.wire_set.add(name)
        g_node = set(g.graph.nodes())
        interset = g_node & self.wire_set
        ll = len(interset)
        while(len(interset)!=0):
            pre_len = len(interset)
            g = self.eliminate_wire(g)
            g_node = set(g.graph.nodes())
            interset = g_node & self.wire_set
            post_len = len(interset)
            if pre_len == post_len:
                break
        if len(interset) != 0:
            print('Warning: uneliminated wire: ', len(interset))
            for n in interset.copy():
                neighbor = self.graph.get_neighbors(n)
                if len(neighbor) == 0:
                    self.graph.remove_node(n)
                    interset.remove(n)

            print('Final uneliminated wire: ', len(interset))
        else:
            print('Finish!\n')
        node_dict = self.graph.node_dict.copy()
        self.graph = g
        self.graph.load_node_dict(node_dict)

    def eliminate_wire(self, g:Graph):
        node_set = g.get_all_nodes()
        for node in node_set:
            node_list = g.get_neighbors(node)
            if node in self.wire_set:
                self.wire_dict[node] = node_list
            else:
                self.temp_dict[node] = node_list
        g_new = Graph()
        for node, node_list in self.temp_dict.items():
            for n in node_list:
                if n in self.wire_dict.keys():
                    wire_assign = self.wire_dict[n]
                    for w in wire_assign:
                        if w:
                            g_new.add_edge(node, w, w)
                else:
                    g_new.add_edge(node, n)
        return g_new



    def subgraph2expr(self, subgraph, cur_node, in_out='In', is_dff=False):
        ## detect cycle
        
        if in_out == "Out":
            out_node_dict = {}
            for n in subgraph.nodes():
                out_node_dict[n] = self.graph.node_dict[n].tpe
            return out_node_dict
        
        smt_vars = {}
        if is_dff:
            cur_node = cur_node + '_'
        smt_vars[cur_node] = Symbol(cur_node, BOOL)
        x_idx = 0
        if not nx.is_directed_acyclic_graph(subgraph):
                # print('WARNING: cycle!')
            traverse_lst = [cur_node]
        else:
            traverse_lst = nx.topological_sort(subgraph)
         
        for n in traverse_lst:
            node = self.graph.node_dict[n]
            if in_out == 'In':
                pred_lst = [i if subgraph.has_edge(i, n) else None for i in subgraph.predecessors(n)]
            else:
                pred_lst = [i if subgraph.has_edge(n, i) else None for i in subgraph.successors(n)]
            smt_vars[n] = Symbol(n, BOOL)
            for n_p in pred_lst:
                if n_p not in smt_vars.keys():
                    smt_vars[n_p] = Symbol(n_p, BOOL)
            if (not node.tpe) or (node.tpe in ['Input', 'Output', 'DFF', 'DLH']) or (pred_lst == []):
                continue
            elif node.tpe in ['INV','BUF']:
                if node.tpe == 'INV':
                    smt_vars[n] = Not(smt_vars[pred_lst[0]])
                elif node.tpe == 'BUF':
                    smt_vars[n] = smt_vars[pred_lst[0]]
            elif node.tpe in ['AND', 'NAND', 'XNOR', 'OR', 'NOR', 'XOR']:
                if len(pred_lst) == 1:
                    smt_vars[f'x_{x_idx}'] = Symbol(f"x_{x_idx}", BOOL)
                    pred_lst.append(f'x_{x_idx}')
                    x_idx += 1
                elif len(pred_lst) > 2:
                    pred_lst = [pred_lst[0], pred_lst[1]]
                assert len(pred_lst) == 2
                ### len(pred_lst) is 2
                if node.tpe == 'AND':
                    smt_vars[n] = And([smt_vars[pred_lst[0]], smt_vars[pred_lst[1]]])
                elif node.tpe == 'NAND':
                    smt_vars[n] = Not(And([smt_vars[pred_lst[0]], smt_vars[pred_lst[1]]]))
                elif node.tpe == 'XNOR':
                    smt_vars[n] = Not(Xor(smt_vars[pred_lst[0]], smt_vars[pred_lst[1]]))
                elif node.tpe == 'OR':
                    smt_vars[n] = Or([smt_vars[i] for i in subgraph.predecessors(n)])
                elif node.tpe == 'NOR':
                    smt_vars[n] = Not(Or([smt_vars[i] for i in subgraph.predecessors(n)]))
                elif node.tpe == 'XOR':
                    smt_vars[n] = Xor(smt_vars[pred_lst[0]], smt_vars[pred_lst[1]])
            elif node.tpe in ['MUX', 'HA', 'FA']:
                if len(pred_lst) == 1:
                    smt_vars[f'x_{x_idx}'] = Symbol(f"x_{x_idx}", BOOL)
                    pred_lst.append(f'x_{x_idx}')
                    x_idx += 1
                    smt_vars[f'x_{x_idx}'] = Symbol(f"x_{x_idx}", BOOL)
                    pred_lst.append(f'x_{x_idx}')
                    x_idx += 1
                elif len(pred_lst) == 2:
                    smt_vars[f'x_{x_idx}'] = Symbol(f"x_{x_idx}", BOOL)
                    pred_lst.append(f'x_{x_idx}')
                    x_idx += 1
                elif len(pred_lst) > 3:
                    pred_lst = [pred_lst[0], pred_lst[1], pred_lst[2]]
                assert len(pred_lst) == 3
                ### len(pred_lst) is 3
                if node.tpe == 'MUX':
                    smt_vars[n] = Ite(smt_vars[pred_lst[2]], smt_vars[pred_lst[1]], smt_vars[pred_lst[0]])
                elif node.tpe == 'HA':
                    smt_vars[n] = And(smt_vars[pred_lst[0]], smt_vars[pred_lst[1]])
                elif node.tpe == 'FA':
                    smt_vars[n] = Or(And(smt_vars[pred_lst[0]], smt_vars[pred_lst[1]]), And(Or(smt_vars[pred_lst[0]], smt_vars[pred_lst[1]]), smt_vars[pred_lst[2]]))
            elif node.tpe in ['AOI', 'OAI']:
                if len(pred_lst) == 1:
                    smt_vars[f'x_{x_idx}'] = Symbol(f"x_{x_idx}", BOOL)
                    pred_lst.append(f'x_{x_idx}')
                    x_idx += 1
                    smt_vars[f'x_{x_idx}'] = Symbol(f"x_{x_idx}", BOOL)
                    pred_lst.append(f'x_{x_idx}')
                    x_idx += 1
                elif len(pred_lst) == 2:
                    smt_vars[f'x_{x_idx}'] = Symbol(f"x_{x_idx}", BOOL)
                    pred_lst.append(f'x_{x_idx}')
                    x_idx += 1

                if node.tpe == 'AOI':
                    if len(pred_lst) == 4:
                        smt_vars[n] = Not(Or(And(smt_vars[pred_lst[3]], smt_vars[pred_lst[2]]), And(smt_vars[pred_lst[1]], smt_vars[pred_lst[0]])))
                    elif len(pred_lst) == 3:
                        smt_vars[n] = Or(And(smt_vars[pred_lst[0]], smt_vars[pred_lst[1]]), smt_vars[pred_lst[2]])
                elif node.tpe == 'OAI':
                    if len(pred_lst) == 4:
                        smt_vars[n] = Not(And(Or(smt_vars[pred_lst[3]], smt_vars[pred_lst[2]]), Or(smt_vars[pred_lst[1]], smt_vars[pred_lst[0]])))
                    elif len(pred_lst) == 3:
                        smt_vars[n] = And(Or(smt_vars[pred_lst[0]], smt_vars[pred_lst[1]]), smt_vars[pred_lst[2]])
            else:
                print('ERROR: New Situation!')
                print(node.tpe)
                assert False

        if is_dff:
            expr = smt_vars[n]
        else:
            expr = smt_vars[cur_node]
        return expr
    
    
    def remove_reg_loop(self, g:nx.DiGraph):
        ### split each reg node into two nodes, with one node connect all the predecessors of reg node, and the other node connect all the successors of reg node
        g_cp = copy.deepcopy(g)
        for n in g.nodes():
            if self.graph.node_dict[n].tpe == 'DFF':
                node = self.graph.node_dict[n]
                ## add new node for predecessors
                new_node_pre = self.graph.node_dict[n].name + '_'
                node_pre = copy.deepcopy(node)
                node_pre.name = new_node_pre
                predecessors = g_cp.predecessors(n)
                g_cp.add_node(new_node_pre)
                self.graph.node_dict[new_node_pre] = node_pre
                for pre in predecessors:
                    g_cp.add_edge(pre, new_node_pre)
                ## add new node for successors
                new_node_suc = self.graph.node_dict[n].name + '__'
                node_suc = copy.deepcopy(node)
                node_suc.name = new_node_suc
                successors = g_cp.successors(n)
                g_cp.add_node(new_node_suc)
                self.graph.node_dict[new_node_suc] = node_suc
                for suc in successors:
                    g_cp.add_edge(new_node_suc, suc)
                ## remove reg node
                g_cp.remove_node(n)

        return g_cp


    def bfs_backtrace_dff(self, param):
        """
        Perform BFS to backtrace all the predecessors of a DFF node until reaching an input node or another DFF node.

        :param start_node: The starting DFF node.
        :param is_input_node: Function to check if a node is an input node.
        :param is_dff_node: Function to check if a node is a DFF node.
        :param get_predecessors: Function to get the predecessors of a node.
        :return: List of nodes visited during the BFS.
        """
        start_DFF, save_dir, design_name = param
        g = nx.DiGraph(self.graph.graph)
        node_dict = self.graph.node_dict
        queue = deque([start_DFF])
        visited = set([start_DFF])
        result = []
        while queue:
            current_node = queue.popleft()
            for predecessor in list(g.predecessors(current_node)):
                node = node_dict[predecessor]
                if node.tpe in ['DFF', 'Input']:
                    result.append(predecessor)
                    continue
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)
                    result.append(predecessor)
        dff_subgraph = g.subgraph(result)
        dff_subgraph = nx.DiGraph(dff_subgraph)
        # start_DFF = re.sub(r"^\\", "", start_DFF)
        start_DFF = re.sub(r"/", "_", start_DFF)
        with open(f"{save_dir}/{design_name}/{start_DFF}.pkl", 'wb') as f:
            pickle.dump(dff_subgraph, f)
        print(dff_subgraph)

        return dff_subgraph

    def update_one_node(self, param):
        subg, node, k = param
        node_dict = self.graph.node_dict
        if node_dict[node].in_expr:
            return
        if node_dict[node].tpe not in [None, "Input"]:
            k_hop_subgraph = nx.ego_graph(subg, node, radius=k, undirected=True)


            input_subgraph = subg.subgraph(
                [n for n in k_hop_subgraph if nx.has_path(subg, n, node)]
            )
            
            output_subgraph = subg.subgraph(
                [n for n in k_hop_subgraph if nx.has_path(subg, node, n)]
            )
            
            in_expr = self.subgraph2expr(input_subgraph, node, 'In', is_dff=False)
            out_expr = self.subgraph2expr(output_subgraph, node, 'Out', is_dff=False)
            node_dict[node].in_expr = in_expr
            node_dict[node].out_expr = out_expr

        return

    

    def update_ppa_info(self, design_name):
        pt_info_dir = f"../../../data_collect/data_pt/init//{design_name}"
        with open (f"{pt_info_dir}/cell.json", 'r') as f:
            cell_dict_all = json.load(f)
        with open (f"{pt_info_dir}/cell_delay.json", 'r') as f:
            cell_delay_dict_all = json.load(f)
        with open (f"{pt_info_dir}/net.json", 'r') as f:
            net_dict_all = json.load(f)
        with open (f"{pt_info_dir}/net_delay.json", 'r') as f:
            net_delay_dict_all = json.load(f)

        label_info_dir = f"../../../data_collect/data_pt/place/{design_name}"
        with open (f"{label_info_dir}/cell.json", 'r') as f:
            label_cell_dict_all = json.load(f)
        with open (f"{label_info_dir}/cell_delay.json", 'r') as f:
            label_delay_dict_all = json.load(f)
        with open (f"{label_info_dir}/slack.json", 'r') as f:
            label_slack_dict_all = json.load(f)

        feat_cell = set(cell_dict_all.keys())
        label_cell = set(label_cell_dict_all.keys())
        no_intersect = feat_cell - label_cell
        intersect = feat_cell & label_cell
        # print('No intersect: ', no_intersect)
        print('No intersect: ', len(no_intersect))
        print('Intersect: ', len(intersect))
        print('Total: ', len(feat_cell), len(label_cell))
        print(len(no_intersect)/len(feat_cell))

        g_nx = nx.DiGraph(self.graph_wire.graph)
        node_dict = self.graph.node_dict

        for name, _ in node_dict.copy().items():
            name_dct = re.sub(r"^\\", "", name)
            ### update feature
            if name_dct in cell_dict_all:
                cell_dict = cell_dict_all[name_dct]
                scale = 1000000
                node = self.graph.node_dict[name]
                node.inter_pwr = round(cell_dict['inter_pwr']*scale,3)
                node.swith_pwr = round(cell_dict['switch_pwr']*scale,3)
                node.leak_pwr = round(cell_dict['leak_pwr']*scale,3)
                node.pwr = round(cell_dict['cell_pwr']*scale,3)
                node.area = round(cell_dict['cell_area'],3)

            if name_dct in cell_delay_dict_all:
                cell_delay_dict = cell_delay_dict_all[name_dct]
                node = self.graph.node_dict[name]
                node.delay = round(cell_delay_dict,3)

            if name_dct in net_dict_all:
                net_dict = net_dict_all[name_dct]
                if name not in g_nx:
                    continue
                for n in g_nx.successors(name):
                    node = self.graph.node_dict[n]
                    node.load = round(net_dict['net_load']*100,3)
                    node.tr = round(net_dict['net_tr'],3)
                    node.prob = round(net_dict['net_prob'],3)

            if name_dct in net_delay_dict_all:
                net_delay_dict = net_delay_dict_all[name_dct]
                if name not in g_nx:
                    continue
                for n in g_nx.successors(name):
                    node = self.graph.node_dict[n]
                    node.cap = round(net_delay_dict['cap'],3)
                    node.res = round(net_delay_dict['res']*100,3)

            ### update label
            if name_dct in label_cell_dict_all:
                cell_dict = label_cell_dict_all[name_dct]
                scale = 1000000
                node = self.graph.node_dict[name]
                node.label_pwr = round(cell_dict['cell_pwr']*scale,3)
                node.label_area = round(cell_dict['cell_area'],3)
            
            if name_dct in label_delay_dict_all:
                cell_delay_dict = label_delay_dict_all[name_dct]
                node = self.graph.node_dict[name]
                node.label_delay = round(cell_delay_dict,3)
            
            if name_dct in label_slack_dict_all:
                slack_dict = label_slack_dict_all[name_dct]
                node = self.graph.node_dict[name]
                node.slack = round(slack_dict,3)


    def graph_split(self, save_dir, design_name):
        g_nx = nx.DiGraph(self.graph.graph)
        print('----- Updating PPA Info -----')
        for n in g_nx.nodes():
            node = self.graph.node_dict[n]
            node.label_pwr = None
            node.label_area = None
            node.label_delay = None
            node.slack = None
        self.update_ppa_info(design_name)


        DFF_lst = []
        g_nx = nx.DiGraph(self.graph.graph)
        node_dict = self.graph.node_dict
        for n in g_nx.nodes():
            if node_dict[n].tpe == 'DFF':
                DFF_lst.append(n)
        print(len(DFF_lst))
        print('----- Splitting Graph -----')
        if not os.path.exists(f"{save_dir}/{design_name}"):
            os.makedirs(f"{save_dir}/{design_name}")
        
        param_lst = []
        for dff in DFF_lst:
            param = (dff, save_dir, design_name)
            param_lst.append((dff, save_dir, design_name))
            subgraph = self.bfs_backtrace_dff(param)
            self.subgraph_annotation(subgraph)

        with open (f"{save_dir}/{design_name}/{design_name}_node_dict.pkl", 'wb') as f:
            pickle.dump(self.graph.node_dict, f)
        
        for n in g_nx.nodes():
            node = self.graph.node_dict[n]
            print(n, node)
            print(f"label: {node.label_pwr}, {node.label_area}, {node.label_delay}, {node.slack}")

    def subgraph_annotation(self, subgraph):
        k = 3
        for n in subgraph.nodes():
            self.update_one_node((subgraph, n, k))
            
        

        
        
            