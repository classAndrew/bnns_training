#ifndef ANDREW_BNNS_DENSE
#define ANDREW_BNNS_DENSE

#include <vecLib/BNNS/bnns.h>
#include "adam_optim.h"

typedef struct {
    float *input_buffer; 
    BNNSNDArrayDescriptor weights_desc;
    BNNSNDArrayDescriptor bias_desc;
    BNNSNDArrayDescriptor in_desc;
    BNNSNDArrayDescriptor out_desc;
    BNNSNDArrayDescriptor in_delta_desc;
    BNNSNDArrayDescriptor weights_delta_desc;
    BNNSNDArrayDescriptor bias_delta_desc;
    BNNSLayerParametersFullyConnected layer_params;
    BNNSFilterParameters filter_params;
    BNNSFilter filter;
    AdamOptimizer weight_optim;
    AdamOptimizer bias_optim;
} DenseLayer;

DenseLayer dense_layer(size_t m, size_t n);
int dense_forward(DenseLayer *dense, float *input, float *output);
int dense_backward(DenseLayer *dense, float *dense_output, const BNNSNDArrayDescriptor *dense_output_delta_desc);
int dense_adam_optim(DenseLayer *dense);

#endif