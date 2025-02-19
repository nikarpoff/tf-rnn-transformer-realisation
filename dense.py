# Copyright [2024] [Nikita Karpov]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, units: int, input_length: int, use_bias: bool, seed: int, name=None):
        super().__init__(name)

        self.units = units

        self.W = tf.Variable(tf.initializers.GlorotUniform(seed=seed)(shape=(input_length, units)), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros(units, dtype=tf.float32), trainable=use_bias)

    def __call__(self, x):
        return tf.sigmoid(tf.matmul(x, self.W) + self.b)
    
    def __str__(self):
        return f"Dense. Output shape: (None, {self.units})"
