#include "bnns_training/mlp.h"
#include <stdlib.h>
#include <vecLib/BNNS/bnns.h>

int MLP_forward(size_t n);
int MLP_backward(MSELossLayer *mse_loss, float *target, float *result);

// add option for loss type and activation function type
MLPnet* MLP_network(size_t n_layers, float *layer_dims);

void MLP_destroy(MLPnet *net) {
    // destroy dense layers
    for (int i = 0; i < net->n_layers; i++) {
        // free all the descs
        free(net->layers[i].weights_desc.data);
        free(net->layers[i].bias_desc.data);
        free(net->layers[i].in_delta_desc.data);
        free(net->layers[i].weights_delta_desc.data);
        free(net->layers[i].bias_delta_desc.data);

        // free all the optims
        free(net->layers[i].bias_optim.adam_acc1_desc.data);
        free(net->layers[i].bias_optim.adam_acc2_desc.data);
        free(net->layers[i].weight_optim.adam_acc1_desc.data);
        free(net->layers[i].weight_optim.adam_acc2_desc.data);

        free(net->layers[i].input_buffer);
        BNNSFilterDestroy(net->layers[i].filter);
    }

    // free the loss layer
    free(net->loss_layer->in_delta_desc.data);
    free(net->loss_layer->out_delta_desc.data);
    BNNSFilterDestroy(net->loss_layer->filter);

    // free output layer
    free(net->last_layer->input_buffer);

    // free layer dims
    free(net->layer_dims);

    // free allocated field structs
    free(net->loss_layer);
    free(net->last_layer);
    free(net->layers);

    // free parent struct
    free(net);
}
