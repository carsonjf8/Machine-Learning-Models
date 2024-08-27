from cftron.compute_graph import ComputationGraph

import numpy as np
from enum import Enum

class ComputationGraph:
    NodeType = Enum('NodeType', ['INPUT', 'OPERATION', 'NONE'])
    Operation = Enum('Operation', ['NONE', 'NEG', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'MATMUL',
                                   'CLIP', 'EXP', 'LOG', 'POWER', 'SUM', 'MAX', 'MIN',
                                   'EXPAND_DIMS', 'SQUEEZE'])

    def __init__(self):
        """
        Class constructor.
        """
        self.graph = {}
    
    def create_node_dict(self,
                    type: ComputationGraph.NodeType=NodeType.NONE,
                    step: int=-1,
                    operation: ComputationGraph.Operation=Operation.NONE,
                    to_node: str=None,
                    from_node_1: str=None,
                    from_node_2: str=None,
                    output_data: np.array=None,
                    gradient: np.array|None=None,
                    op_args: dict|None=None) -> dict:
        """
        Creates a dictionary of values describing an operation.

        Parameters
        ----------
        type : ComputationGraph.NodeType, default = NodeType.NONE
            Enum describing whether the node is an operation or an input.
        step : int, default= - 1
            Operation order number, zero ordered.
        operation : ComputationGraph.Operation, default = Operation.NONE
            Enum representing what operation is described by the node.
        to_node : str, default = None
            Node to which the output data of this node goes to.
        from_node_1 : str, default = None
            Node for which the output from is the input to this node. If binary operation, this is the left input.
        from_node_2 : str, default = None
            Node for which the output from is the right input into the operation of this node.
        output_data : np.array, default = None
            Output result from the operation.
        gradient : np.array | None, default = None
            Gradient of the final value with respect to the output data of this node. Calculated during the backward pass.
        op_args : dict | None, default = None
            Dictionary of arguments passed to the operation.
        
        Returns
        -------
        dict
            Dictionary of keys and values describing an operation in the computation graph.
        """
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

    def create_input_node_dict(self, data: np.array) -> dict:
        """
        Creates and returns a node dictionary describing an input node.

        Parameters
        ----------
        data : np.array
            Input data array.
        
        Returns
        -------
        dict
            Node dictionary describing an input value.
        """
        return self.create_node_dict(
            type=ComputationGraph.NodeType.INPUT,
            step=0,
            operation=ComputationGraph.Operation.NONE,
            output_data=data)

    def add_operation(self,
                      op: ComputationGraph.NodeType,
                      uuid_result: str,
                      data_result: np.array,
                      uuid_1: str,
                      data_1: np.array,
                      uuid_2: str|None=None,
                      data_2: np.array|None=None,
                      op_args: dict|None=None) -> None:
        """
        Adds an operation to the computation graph.

        Parameters
        ----------
        op : ComputationGraph.NodeType
            Enum describing whether the node is an operation or an input.
        uuid_result : str
            UUID of the resulting node.
        data_result : np.array
            Output array from the operation.
        uuid_1 : str
            UUID of the input node to the operation. Left input UUID if the operation is binary.
        data_1 : np.array
            Input data to the operation. Left input data if the operation is binary.
        uuid_2 : str | None, default = None
            UUID of the right input to the operation.
        data_2 : np.array | None, default = None
            Right input data to the operation.
        op_args : dict | None, default = None
            Dictionary of parameters passed to the operation.
        """
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

    def backward(self, uuid: str) -> None:
        """
        Backward pass that calculates the gradients of the inputs.

        Parameters
        ----------
        uuid : str
            UUID of the node to calculate the gradient of.
        """
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

    def reset(self) -> None:
        """
        Clears the computation graph.
        """
        self.graph = {}
    
    def reset_grads(self) -> None:
        """
        Clears all gradient values in the computation graph.
        """
        for key in self.graph:
            self.graph[key]['gradient'] = np.zeros_like(self.graph[key]['output_data'])
