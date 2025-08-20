
from collections import defaultdict
from multiprocessing import Pool
import sys, re, os
import numpy as np
import pickle
import networkx as nx
# sys.setrecursionlimit(10000)

class Node:
    def __init__(self, name, type, width:None, father:None, tpe:None, lineno:None):
        self.name = name
        self.type = type
        self.width = width
        self.father = father
        self.tpe = tpe
        self.subgraph = []
        self.lineno = lineno
        self.node_text = None
        self.in_expr = None
        self.out_expr = None
        
        self.pwr = 0
        self.area = 0
        self.delay = 0
        self.load = 0
        self.tr = 0
        self.prob = 0
        self.cap = 0
        self.res = 0

        self.label_pwr = 0
        self.label_area = 0
        self.label_delay = 0
        self.slack = 0
        self.text_attr = None

    def __repr__(self) -> str:
        repr_str = f'{self.name}: {self.tpe}, {self.lineno}, {self.in_expr}, \n'
        repr_str += f'      physical: {self.pwr}, {self.area}, {self.delay}, {self.load}, {self.tr}, {self.prob}, {self.cap}, {self.res}'
        return repr_str
    


class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_dict = {}

    def init_graph(self, graph, node_dict):
        self.graph = graph
        self.node_dict = node_dict

    def add_decl_node(self, name, type, width=None, father=None, tpe=None, strength=None):
        node = Node(name, type, width, father, tpe, strength)
        self.node_dict[name] = node

    def graph2pkl(self, design_name, cmd):
        folder_dir = '/home/coguest5/netlist_parser/graph_data/{cmd}'
        graph_name = folder_dir + f'/{design_name}_{cmd}.pkl'
        node_dict_name = folder_dir + f'/{design_name}_{cmd}_node_dict.pkl'
        with open(graph_name, 'wb') as f:
            pickle.dump(self.graph, f)
        with open(node_dict_name, 'wb') as f:
            pickle.dump(self.node_dict, f)

    def add_edge(self, u, v, weight=None):
        self.graph.add_edge(u, v, weight=weight)
    
    def remove_node(self, u):
        if u in self.graph.copy():
            del self.graph[u]
    
    def get_neighbors(self, u):
        return self.graph[u]
    
    def get_all_nodes(self):
        return self.graph.nodes()
    
    # def get_all_nodes2(self):
    #     all_nodes = set()
    #     for key, val_list in self.graph.nodes():
    #         all_nodes.add(key)
    #         for var in val_list:
    #             all_nodes.add(var)
    #     return all_nodes

    def load_node_dict(self, node_dict):
        self.node_dict = node_dict
    
    def cal_node_width(self):
        print('----- Calculating Operator Width -----')
        self.nowidth_set = set()
        
        for name, node in self.node_dict.items():
            if not node.width:
                self.nowidth_set.add(name)
        print('No width num: ', len(self.nowidth_set))
        while(len(self.nowidth_set) != 0):
            ll_pre = len(self.nowidth_set)
            for n in self.nowidth_set.copy():
                # print(n)
                # assert n in self.graph.keys()
                if n in self.graph.keys():
                    neighbor = self.graph[n]
                    width = self.get_max_neighbor_wdith(neighbor)
                    self.node_dict[n].update_width(width)
                    if width:
                        self.nowidth_set.remove(n)
            ll_post = len(self.nowidth_set)
        #     if ll_pre == ll_post:
        #         break
        # print(ll_post)
        # print(self.nowidth_set)

    def get_max_neighbor_wdith(self, neighbor):
        width_list = []
        for n in neighbor:
            width_node = self.node_dict.get(n)
            if not width_node:
                return width_node
            else:
                width = width_node.width
                width_list.append(width)
        assert len(neighbor) == len(width_list)
        width = max(width_list)
        return width
    
    def get_stat(self):
        # all_node = self.get_all_nodes2()
        self.seq_set = set()
        self.wire_set = set()
        self.comb_set = set()
        self.in_set = set()
        self.out_set = set()
        type_set = set()
        seq_num = 0
        comb_num = 0
        for name, node in self.node_dict.items():
            ntype = node.type
            if ntype == 'Reg':
                self.seq_set.add(name)
            elif ntype == 'Wire':
                self.wire_set.add(name)
            elif ntype in ['Operator', 'UnaryOperator', 'Concat', 'Repeat']:
                self.comb_set.add(name)
            elif ntype in ['Input']:
                self.in_set.add(name)
            elif ntype in ['Output']:
                self.in_set.add(name)

        for name, node in self.node_dict.items():
            father = node.father
            if father:
                if self.node_dict[father].type == 'Reg':
                    self.seq_set.add(name)
                elif self.node_dict[father].type == 'Wire':
                    self.wire_set.add(name)
                elif self.node_dict[father].type == 'Input':
                    self.in_set.add(name)
                elif self.node_dict[father].type == 'Output':
                    self.out_set.add(name)           
    
    def show_graph(self):
        self.get_stat()
        print('----- Writting Graph Visialization File -----')
        outfile_path = "../img/"
        outfile = outfile_path+"AST_graph.dot"
        top_name = 'test'
        node_set = self.graph.keys()
        pair_set = set()
        for vertice in self.graph.keys():
            node_set.add(vertice)
            val_list = self.get_neighbors(vertice)
            for val in val_list:
                if val:
                    if vertice:
                        val = re.sub(r'\.|\[|\]|\\', r'_', val)
                        vertice = re.sub(r'\.|\[|\]|\\', r'_', vertice)
                        pair = '{0} -> {1}'.format(vertice, val)
                        pair_set.add(pair)

        with open (outfile, 'w') as f:
            line = "digraph {0} ".format(top_name)
            line = line + "{\n"
            f.write(line)
            reg_set = set()
            for node in node_set:
                if not node:
                    break
                n = self.node_dict[node]
                ntype = n.type
                node1 = re.sub(r'\.|\[|\]|\\', r'_', node)
                if node in self.seq_set:
                    line = "    {0} [style=filled, color=lightblue];\n".format(node1)
                elif node in self.wire_set:
                    line = "    {0} [style=filled, color=red];\n".format(node1)
                elif node in self.in_set:
                    line = "    {0} [style=filled, color=black];\n".format(node1)
                elif node in self.out_set:
                    line = "    {0} [style=filled, color=green];\n".format(node1)
                elif ntype == 'Constant':
                    line = "    {0} [style=filled, color=grey];\n".format(node1)
                elif node in self.comb_set:
                    line = "    {0} [style=filled, color=pink];\n".format(node1)

                else:
                    line = "    {0};\n".format(node1)
                f.write(line)
            for pair in pair_set:
                line = "    {0};\n".format(pair)
                f.write(line)
            
            f.write("}\n")
        
        print('Finish!\n')




                

