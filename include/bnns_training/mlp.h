#ifndef ANDREW_BNNS_DENSE_LAYER
#define ANDREW_BNNS_DENSE_LAYER

#include "bnns_training/dense_layer.h"
#include "bnns_training/adam_optim.h"
#include "bnns_training/output_layer.h"
#include "bnns_training/loss_layer.h"

typedef struct {
    size_t n_layers;
    float *layer_dims;
    DenseLayer *layers;
    MSELossLayer *loss_layer;
    OutputLayer *last_layer;
} MLPnet;

int MLP_forward(size_t n);
int MLP_backward(MSELossLayer *mse_loss, float *target, float *result);

// add option for loss type and activation function type
MLPnet* MLP_network(size_t n_layers, float *layer_dims);
void MLP_destroy(MLPnet *net);

#endif