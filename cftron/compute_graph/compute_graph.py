import numpy as np
from enum import Enum

class ComputationGraph:
    NodeType = Enum('NodeType', ['INPUT', 'OPERATION', 'NONE'])
    Operation = Enum('Operation', ['NONE', 'NEG', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'MATMUL',
                                   'CLIP', 'EXP', 'LOG', 'POWER', 'SUM', 'MAX', 'MIN',
                                   'EXPAND_DIMS', 'SQUEEZE'])

    def __init__(self):
        self.graph = {}
    
    def create_node_dict(self,
                    type=NodeType.NONE,
                    step=-1,
                    operation=Operation.NONE,
                    to_node=None,
                    from_node_1=None,
                    from_node_2=None,
                    output_data=None,
                    gradient=None,
                    op_args=None):
        return {
            'type': type,
            'step': step,
            'operation': operation,
            'to_node': [] if to_node == None else to_node,
            'from_node_1': from_node_1,
            'from_node_2': from_node_2,
            'output_data': output_data,
            'gradient': gradient if gradient else np.zeros_like(output_data),
            'op_args': op_args
        }

    def create_input_node_dict(self, data):
        return self.create_node_dict(
            type=ComputationGraph.NodeType.INPUT,
            step=0,
            operation=ComputationGraph.Operation.NONE,
            output_data=data)

    def add_operation(self, op, uuid_result, data_result, uuid_1, data_1, uuid_2=None, data_2=None, op_args=None):
        if uuid_1 not in self.graph:
            self.graph[uuid_1] = self.create_input_node_dict(data_1)
        if uuid_2 != None and uuid_2 not in self.graph:
            self.graph[uuid_2] = self.create_input_node_dict(data_2)
        
        self.graph[uuid_result] = self.create_node_dict(
            type=ComputationGraph.NodeType.OPERATION,
            step=self.graph[uuid_1]['step'] + 1 if uuid_2 == None else max(self.graph[uuid_1]['step'], self.graph[uuid_2]['step']) + 1,
            operation=op,
            from_node_1=uuid_1,
            from_node_2=uuid_2,
            output_data=data_result,
            op_args=op_args)
        self.graph[uuid_1]['to_node'].append(uuid_result)
        if uuid_2 != None:
            self.graph[uuid_2]['to_node'].append(uuid_result)

    def backward(self, uuid):
        cur_node = self.graph[uuid]
        if np.isnan(cur_node['gradient']).any() or np.isinf(cur_node['gradient']).any():
            print(np.isnan(cur_node['gradient']).any(), np.isinf(cur_node['gradient']).any())
            print(uuid, cur_node)

        if cur_node['type'] == ComputationGraph.NodeType.INPUT:
            return

        input_1_node = self.graph[cur_node['from_node_1']]
        if cur_node['from_node_2'] is not None:
            input_2_node = self.graph[cur_node['from_node_2']]

        if cur_node['operation'] == ComputationGraph.Operation.NEG:
            input_1_node['gradient'] += -cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.ADD:
            if input_1_node['output_data'].shape != cur_node['gradient'].shape:
                unequal_axis = 0 if input_1_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                input_1_node['gradient'] += np.sum(cur_node['gradient'], axis=unequal_axis).reshape(input_1_node['output_data'].shape)
                input_2_node['gradient'] += cur_node['gradient']
            else:
                if input_2_node['output_data'].shape != cur_node['gradient'].shape:
                    unequal_axis = 0 if input_2_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                    input_1_node['gradient'] += cur_node['gradient']
                    input_2_node['gradient'] += np.sum(cur_node['gradient'], axis=unequal_axis).reshape(input_2_node['output_data'].shape)
                else:
                    input_1_node['gradient'] += cur_node['gradient']
                    input_2_node['gradient'] += cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.SUBTRACT:
            if input_1_node['output_data'].shape != cur_node['gradient'].shape:
                unequal_axis = 0 if input_1_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                input_1_node['gradient'] += np.sum(cur_node['gradient'], axis=unequal_axis).reshape(input_1_node['output_data'].shape)
                input_2_node['gradient'] += -cur_node['gradient']
            else:
                if input_2_node['output_data'].shape != cur_node['gradient'].shape:
                    unequal_axis = 0 if input_2_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                    input_1_node['gradient'] += cur_node['gradient']
                    input_2_node['gradient'] += -np.sum(cur_node['gradient'], axis=unequal_axis).reshape(input_2_node['output_data'].shape)
                else:
                    input_1_node['gradient'] += cur_node['gradient']
                    input_2_node['gradient'] += -cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.MULTIPLY:
            if input_1_node['output_data'].shape != cur_node['gradient'].shape:
                unequal_axis = 0 if input_1_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                input_1_node['gradient'] += np.sum(input_2_node['output_data'] * cur_node['gradient'], axis=unequal_axis).reshape(input_1_node['output_data'].shape)
                input_2_node['gradient'] += input_1_node['output_data'] * cur_node['gradient']
            else:
                if input_2_node['output_data'].shape != cur_node['gradient'].shape:
                    unequal_axis = 0 if input_2_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                    input_1_node['gradient'] += input_2_node['output_data'] * cur_node['gradient']
                    input_2_node['gradient'] += np.sum(input_1_node['output_data'] * cur_node['gradient'], axis=unequal_axis).reshape(input_2_node['output_data'].shape)
                else:
                    input_1_node['gradient'] += input_2_node['output_data'] * cur_node['gradient']
                    input_2_node['gradient'] += input_1_node['output_data'] * cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.DIVIDE:
            if input_1_node['output_data'].shape != cur_node['gradient'].shape:
                unequal_axis = 0 if input_1_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                input_1_node['gradient'] += np.sum(input_2_node['output_data'] / input_2_node['output_data']**2 * cur_node['gradient'], unequal_axis).reshape(input_1_node['output_data'].shape)
                input_2_node['gradient'] += -input_1_node['output_data'] / input_2_node['output_data']**2 * cur_node['gradient']
            else:
                if input_2_node['output_data'].shape != cur_node['gradient'].shape:
                    unequal_axis = 0 if input_2_node['output_data'].shape[0] != cur_node['gradient'].shape[0] else 1
                    input_1_node['gradient'] += input_2_node['output_data'] / input_2_node['output_data']**2 * cur_node['gradient']
                    input_2_node['gradient'] += np.sum(-input_1_node['output_data'] / input_2_node['output_data']**2 * cur_node['gradient'], unequal_axis).reshape(input_2_node['output_data'].shape)
                else:
                    input_1_node['gradient'] += input_2_node['output_data'] / input_2_node['output_data']**2 * cur_node['gradient']
                    input_2_node['gradient'] += -input_1_node['output_data'] / input_2_node['output_data']**2 * cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.MATMUL:
            self.graph[cur_node['from_node_1']]['gradient'] += cur_node['gradient'] @ self.graph[cur_node['from_node_2']]['output_data'].T
            self.graph[cur_node['from_node_2']]['gradient'] += self.graph[cur_node['from_node_1']]['output_data'].T @ cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.CLIP:
            self.graph[cur_node['from_node_1']]['gradient'] += (
                np.ones_like(self.graph[cur_node['from_node_1']]['output_data'])
                * ((self.graph[cur_node['from_node_1']]['output_data'] >= cur_node['op_args']['a_min']) if cur_node['op_args']['a_min'] else True)
                * ((self.graph[cur_node['from_node_1']]['output_data'] <= cur_node['op_args']['a_max']) if cur_node['op_args']['a_max'] else True)
                * cur_node['gradient'])
        elif cur_node['operation'] == ComputationGraph.Operation.EXP:
            self.graph[cur_node['from_node_1']]['gradient'] += np.exp(self.graph[cur_node['from_node_1']]['output_data']) * cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.LOG:
            self.graph[cur_node['from_node_1']]['gradient'] += cur_node['gradient'] / self.graph[cur_node['from_node_1']]['output_data']
        elif cur_node['operation'] == ComputationGraph.Operation.POWER:
            self.graph[cur_node['from_node_1']]['gradient'] += cur_node['op_args']['exp'] * np.power(self.graph[cur_node['from_node_1']]['output_data'], cur_node['op_args']['exp'] - 1) * cur_node['gradient']
        elif cur_node['operation'] == ComputationGraph.Operation.SUM:
            self.graph[cur_node['from_node_1']]['gradient'] += np.ones_like(self.graph[cur_node['from_node_1']]['output_data']) * (np.expand_dims(cur_node['gradient'], axis=cur_node['op_args']['axis']) if cur_node['op_args']['axis'] != None else cur_node['gradient'])
        elif cur_node['operation'] == ComputationGraph.Operation.MAX:
            self.graph[cur_node['from_node_1']]['gradient'] += np.ones_like(self.graph[cur_node['from_node_1']]['output_data']) * (self.graph[cur_node['from_node_1']]['output_data'] == (np.expand_dims(cur_node['output_data'], axis=cur_node['op_args']['axis']) if cur_node['op_args']['axis'] != None else cur_node['output_data'])) * (np.expand_dims(cur_node['gradient'], axis=cur_node['op_args']['axis']) if cur_node['op_args']['axis'] != None else cur_node['gradient'])
        elif cur_node['operation'] == ComputationGraph.Operation.MIN:
            self.graph[cur_node['from_node_1']]['gradient'] += np.ones_like(self.graph[cur_node['from_node_1']]['output_data']) * (self.graph[cur_node['from_node_1']]['output_data'] == (np.expand_dims(cur_node['output_data'], axis=cur_node['op_args']['axis']) if cur_node['op_args']['axis'] != None else cur_node['output_data'])) * (np.expand_dims(cur_node['gradient'], axis=cur_node['op_args']['axis']) if cur_node['op_args']['axis'] != None else cur_node['gradient'])
        elif cur_node['operation'] == ComputationGraph.Operation.EXPAND_DIMS:
            self.graph[cur_node['from_node_1']]['gradient'] += np.squeeze(cur_node['gradient'], axis=cur_node['op_args']['axis'])
        elif cur_node['operation'] == ComputationGraph.Operation.SQUEEZE:
            self.graph[cur_node['from_node_1']]['gradient'] += np.expand_dims(cur_node['gradient'], axis=cur_node['op_args']['axis'])

        if cur_node['step'] - 1 == self.graph[cur_node['from_node_1']]['step']:
            self.backward(cur_node['from_node_1'])
        if cur_node['from_node_2'] != None:
            if cur_node['step'] - 1 == self.graph[cur_node['from_node_2']]['step']:
                self.backward(cur_node['from_node_2'])

    def reset(self):
        self.graph = {}
    
    def print_graph(self):
        raise NotImplementedError
        keys = list(self.graph.keys())
        next_to_print = []
        for k in keys:
            if self.graph[k]['type'] == ComputationGraph.NodeType.INPUT:
                next_to_print.append(k)
        while len(next_to_print) > 0:
            print(keys[0])
            #for k in self.graph[keys[0]]:
                #print(f'"{k}": {self.graph[keys[0]][k]}')
            keys = keys[1:]
            print()

            for node_uuid in self.graph[keys[0]]['to_node']:
                keys.append(node_uuid)
    