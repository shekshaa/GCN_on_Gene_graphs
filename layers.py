from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer by chebyshev polynomials"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., is_cheby=True, sparse_inputs=False, act=tf.nn.relu, bias=False,
                 locality=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.locality = locality
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.is_cheby = is_cheby
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            if is_cheby:
                for i in range(self.locality):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i))
            else:
                for i in range(len(self.support)):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i))
                if self.bias:
                    self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()

        if self.is_cheby:
            for i in range(self.locality):
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            output = tf.add_n(supports)
        else:
            for i in range(len(self.support)):
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']

        self.output = self.act(output)
        return self.output


class SumLayer(Layer):
    def __init__(self, **kwargs):
        super(SumLayer, self).__init__(**kwargs)

    def _call(self, inputs):
        x = inputs
        return tf.reduce_mean(x, axis=0, keepdims=True)


class InceptionGC(Layer):
    def __init__(self, input_dim, output_dim, locality_sizes, placeholders, is_pool=False, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(InceptionGC, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.locality_sizes = locality_sizes
        self.is_pool = is_pool
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # initialization of filters
        with tf.variable_scope(self.name + '_vars'):
            for j in range(len(self.locality_sizes)):
                for i in range(len(self.support)):
                    self.vars['gcn_' + str(j) + '_weights_' + str(i)] = glorot([input_dim, output_dim],
                                                                               name='gcn_' + str(j) + '_weights_' +
                                                                                    str(i))
                if self.bias:
                    self.vars['gcn_' + str(j) + '_bias'] = zeros([output_dim], name='gcn_' + str(j) + '_bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        self.outputs = []
        for l in range(len(self.locality_sizes)):
            supports = []
            for i in range(self.locality_sizes[l] + 1):
                pre_sup = dot(x, self.vars['gcn_' + str(l) + '_weights_' + str(i)],
                              sparse=self.sparse_inputs)
                support = dot(self.support[i], pre_sup, sparse=True)
                # collecting results of different degrees
                supports.append(support)

            # adding results of all degrees as output of one GC layer
            output = tf.add_n(supports)
            # bias
            if self.bias:
                output += self.vars['bias']

            # collecting outputs of GC layers
            self.outputs.append(self.act(output))

        # in case of pooling
        if self.is_pool:
            # stacking up outputs of all GC layers in a block
            aux = tf.stack(self.outputs, 1)

            # apply max-out over different GC layers
            aux2 = tf.nn.pool(aux, [len(self.locality_sizes)], pooling_type='MAX', padding='VALID', data_format='NWC')

            # removing dimensions of size 1
            aux2 = tf.squeeze(aux2, [1])
            self.pooled_outputs = aux2

        # not pooling just concatenating outputs of all GC layers
        else:
            self.pooled_outputs = tf.concat(self.outputs, 1)

        # final output of GC block
        return self.pooled_outputs
