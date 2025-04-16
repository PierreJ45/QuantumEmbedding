import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

import pennylane as qml
from pennylane import numpy as np


class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]
        self.num_qubits = configs["num_qubits"]
        self.QML_device = configs["QML_device"]
        self.num_layers = configs["num_layers"]
        self.diff_method = configs["diff_method"]

        self.hybrid_qml_model = Hybrid_QML_Model(self.seq_len, self.pred_len, self.num_qubits, self.QML_device, self.num_layers, self.diff_method)     # nn.Linear(self.seq_len, self.pred_len)
        

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.hybrid_qml_model(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]
    


class Hybrid_QML_Model(nn.Module):
    def __init__(self, lookback_window_size, forecast_window_size, num_qubits, QML_device, num_layers, diff_method): 
        super(Hybrid_QML_Model, self).__init__() 
        self.forecast_window_size = forecast_window_size
        
        self.dev = qml.device(QML_device, wires = num_qubits)
        
        @qml.qnode(self.dev, interface = 'torch' , diff_method=diff_method)
        def quantum_function(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        self._quantum_circuit = quantum_function
        
        print("Selected differentiation method:", self._quantum_circuit.gradient_fn)

        q_weights_shape = {'weights':(num_layers, num_qubits, 3)}

        self.input_classical_layer = torch.nn.Linear(lookback_window_size, 2 ** num_qubits) 

        self.hidden_quantum_layer = qml.qnn.TorchLayer(self._quantum_circuit, q_weights_shape)        
        
        self.output_classical_layer = torch.nn.Linear(num_qubits, forecast_window_size) 

    
    def forward(self, batch_input):
        # batch_input: [Batch, Channel, Input length]
        y = batch_input.reshape(batch_input.shape[0] * batch_input.shape[1], batch_input.shape[2])   # Merge the batch and channel dimension

        y = self.input_classical_layer(y) 

        y = self.hidden_quantum_layer(y) 

        y = self.output_classical_layer(y) 

        batch_output = y.reshape(batch_input.shape[0], batch_input.shape[1], self.forecast_window_size)        
        
        return batch_output  # batch_output: [Batch, Channel, Output length]
