import tensorflow as tf

class CrossAttentionUnit(tf.keras.layers.Layer):
    def __init__(self, units: int, token_length: int, seed: int, name=None):
        super().__init__(name=name)
        self.units = units
        self.seed = seed
        self.token_length = token_length

        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def build(self, input_shape):
        # Use Glorot Uniform initializer for weights initializing.
        w_initializer = tf.initializers.GlorotUniform(seed=self.seed)

        # Initialize query matrix weigths.
        self.W_q = self.add_weight(shape=(self.token_length, self.units), initializer=w_initializer, name=f"W_q")
        
        # Initialize key matrix weigths.
        self.W_k = self.add_weight(shape=(self.token_length, self.units), initializer=w_initializer, name=f"W_k")

        # Initialize value matrix weigths.
        self.W_v = self.add_weight(shape=(self.token_length, self.units), initializer=w_initializer, name=f"W_v")

        super().build(input_shape)

    def call(self, x: tf.Tensor, y: tf.Tensor):
        # Compute Q, K, V projections.
        Q = tf.matmul(x, self.W_q)  # (batch, x_len, units)
        V = tf.matmul(y, self.W_v)  # (batch, y_len, units)
        K = tf.matmul(y, self.W_k)  # (batch, y_len, units)

        # Compute attention alignment.
        E = tf.matmul(Q, K, transpose_b=True)  # (batch, x_len, y_len)

        # Compute attention matrix along the K axis (y_len).
        A = self.softmax(E)

        # Compute result
        O = tf.matmul(A, V)

        return O

x = tf.random.normal((64, 10, 300))
y = tf.random.normal((64, 15, 300))

attn = CrossAttentionUnit(units=128, token_length=300, seed=7)
output = attn(x, y)

print(output.shape)
