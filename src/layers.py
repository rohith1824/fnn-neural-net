import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, activation='linear'):
        """
        input_dim:  number of inputs into this layer
        output_dim: number of neurons in this layer
        activation: 'relu', 'sigmoid', or 'linear'
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # ← CHANGE HERE: He initialization for ReLU layers
        # If using ReLU, scale by sqrt(2/input_dim). If linear/sigmoid, use small stddev.
        if activation == 'relu':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        else:
            # for 'linear' or 'sigmoid', a small Gaussian still works
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        # Biases start at zero
        self.b = np.zeros((1, output_dim))

        self.input = None
        self.z = None
        self.output = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.W) + self.b

        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.output = 1.0 / (1.0 + np.exp(-self.z))
        elif self.activation == 'linear':
            self.output = self.z.copy()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        return self.output

    def backward(self, dout):
        # Compute derivative through the activation
        if self.activation == 'relu':
            dz = dout * (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1.0 / (1.0 + np.exp(-self.z))
            dz = dout * sig * (1 - sig)
        elif self.activation == 'linear':
            dz = dout.copy()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        # Gradients w.r.t. W and b
        self.dW = np.dot(self.input.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.W.T)
        return dx

    def __repr__(self):
        return f"Dense(in={self.input_dim}, out={self.output_dim}, act={self.activation})"

