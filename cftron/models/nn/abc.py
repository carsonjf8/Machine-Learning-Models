import numpy as np

class MLP:
    '''
    B: batch size
    D: dimensions of the input data
    C: number of classes (number of nerons in last layer)
    '''

    def __init__(self, n_inputs, layer_n_units, layer_activation_fns, loss_fn, learning_rate, batch_size) -> None:
        self.n_inputs = n_inputs
        self.layer_n_units = layer_n_units
        self.layer_activation_fns = layer_activation_fns
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = None
    
    def train(self, x_train, y_train, epochs, epoch_start=1):
        loss_history = {}

        for epoch in range(epoch_start, epoch_start + epochs):
            #print('epoch', epoch, '/', epoch_start + epochs - 1)
            batch_sample_index = 0
            while batch_sample_index < x_train.shape[0]:
                x_batch = x_train[batch_sample_index : batch_sample_index + self.batch_size]
                y_batch = y_train[batch_sample_index : batch_sample_index + self.batch_size]
                batch_sample_index += self.batch_size

                output = self._forward(x_batch)
                #print('forward output:', output)
                loss = self.loss_fn(output, y_batch)
                #print('loss', loss)
                loss_grad = self.loss_fn(output, y_batch, backward=True)
                self._backward(loss_grad, self.learning_rate)
                #print()
            print(f'EPOCH {epoch}/{epoch_start + epochs - 1}: LOSS: {loss}')
            if 'epoch' in loss_history:
                lh = loss_history['epoch']
                lh.append(epoch)
                loss_history['epoch'] = lh
            else:
                loss_history['epoch'] = [epoch]
            if 'loss' in loss_history:
                lh = loss_history['loss']
                lh.append(loss)
                loss_history['loss'] = lh
            else:
                loss_history['loss'] = [loss]
        return loss_history
    
    '''
    process data through the model
    x_batch: NumPy array
        batch of data inputs, shape: (B, D)
    returns: NumPy array (B, C)
    '''
    def _forward(self, x_batch):
        x = x_batch
        for layer in self.layers:
            #print(x.shape, layer.weights.shape, end=' ')
            x = layer._forward(x)
            #print('end', x.shape)
        return x
    
    def build(self) -> None:
        assert len(self.layer_n_units) == len(self.layer_activation_fns)
        # assemble layers
        self.layers = [Dense(n_units, activation_fn) for (n_units, activation_fn) in zip(self.layer_n_units, self.layer_activation_fns)]
        # build layers
        for index, layer in enumerate(self.layers):
            layer._build(self.layers[index - 1].n_units if index > 0 else self.n_inputs)
    
    def _backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            print('backward', layer.n_units, np.mean(gradient))
            gradient = layer._backward(gradient, learning_rate)
    
    def predict(self, x_batch):
        x = x_batch
        for layer in self.layers:
            x = layer._forward(x)
        y_pred = np.argmax(x, axis=1)
        return y_pred

class Dense:

    '''
    B: batch size
    D: dimensions of the input data
    N: number of neurons in this layer
    P: number of neurons in the previous layer
    '''

    def __init__(self, n_units, activation_fn) -> None:
        self.n_units = n_units
        self.activation_fn = activation_fn
        self.weights = None # shape: (P, N)
        self.biases = None # shape: (1, N)
        self.last_x_batch = None # shape: (B, P)
        self.last_output = None # shape: (B, N)
    
    '''
    process data through the model
    x_batch: NumPy array
        batch of data inputs, shape: (B, P)
    returns: NumPy array
        processed data, shape: (B, N)
    '''
    def _forward(self, x_batch: np.ndarray) -> np.ndarray:
        self.last_x_batch = x_batch # store for use in backprop
        output = x_batch @ self.weights # (B, P) @ (P, N) = (B, N)
        output = output + self.biases # (B, N) + (1, N) = (B, N)
        if self.activation_fn:
            output = self.activation_fn(output) # (B, N)
        self.last_output = output

        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            print('forward', self.n_units)
            print(np.any(np.isnan(output)), np.any(np.isinf(output)))
            print('x_batch', x_batch)
            print('weights', self.weights)
            print('output', output)
            print()
            quit()

        return output # (B, N)

    def _build(self, n_in: list) -> None:
        self.weights = np.random.normal(0, 1, (n_in, self.n_units))# * 2 - 1 # (P, N)
        self.biases = np.random.normal(0, 1, (1, self.n_units))# * 2 - 1 # (1, N)
    
    def _backward(self, dl_dy, learning_rate):
        if np.any(np.isnan(dl_dy)):
            print('enter', self.n_units, dl_dy)

        #print('backward', self.n_units, self.activation_fn, dl_dy, dl_dy.shape)
        # gradient for activation function
        if self.activation_fn:
            dl_dy = np.multiply(dl_dy, self.activation_fn(self.last_output, backward=True)) # (B, N)
        #print('dl_dy.shape (B, N)', dl_dy.shape)

        # compute gradients
        dl_dw = self.last_x_batch.T @ dl_dy # (P, B) @ (B, N) = (P, N))
        #print('dl_dw', dl_dw)
        #print('dl_dw.shape (P, N)', dl_dw.shape)
        dl_db = np.expand_dims(np.sum(dl_dy, axis=0), axis=0) # (1, N) = (1, N)
        #print('dl_db', dl_db)
        #print('dl_db.shape (1, N)', dl_db.shape)
        dl_dx = dl_dy @ self.weights.T # (B, N) @ (N, P) = (B, P)
        if np.any(np.isnan(dl_dx)) or np.any(np.isinf(dl_dx)):
            print('backward', self.n_units)
            print(np.any(np.isnan(dl_dx)), np.any(np.isinf(dl_dx)))
            print('dl_dx', dl_dx)
            print('dl_dy', dl_dy)
            print('weights', self.weights)
            quit()
        
        #print('dl_dx', dl_dx)
        #print('dl_dx.shape (B, P)', dl_dx.shape)

        # save gradients
        self.dl_dy = dl_dy
        self.dl_dw = dl_dw
        self.dl_db = dl_db
        self.dl_dx = dl_dx

        # update weights and biases
        self.weights = self.weights - learning_rate * dl_dw # (P, N) - (1) * (P, N) = (P, N)
        self.biases = self.biases - learning_rate * dl_db # (1, N) - (1) * (1, N)

        if np.any(np.isnan(dl_dy)) or np.any(np.isnan(dl_dw)) or np.any(np.isnan(dl_db)) or np.any(np.isnan(dl_dx)) or np.any(np.isnan(self.weights)) or np.any(np.isnan(self.biases)):
            print('layer backprop', self.n_units, self.activation_fn)
            print('dl_dy', dl_dy)
            print('dl_dw', dl_dw)
            print('dl_db', dl_db)
            print('dl_dx', dl_dx)
            print('weights', self.weights)
            print('biases', self.biases)
            print()
            quit()

        return dl_dx
