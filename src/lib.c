#include <vecLib/BNNS/bnns.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bnns_training/adam_optim.h"
#include "bnns_training/dense_layer.h"
#include "bnns_training/loss_layer.h"
#include "bnns_training/output_layer.h"

void list_init(size_t *target, size_t list[BNNS_MAX_TENSOR_DIMENSION]) {
    for (int i = 0; i < BNNS_MAX_TENSOR_DIMENSION; i++) {
        target[i] = list[i];
    }
}

float *rand_mat(size_t m, size_t n) {
    float *tensor = malloc(sizeof(float) * m * n);
    for (int i = 0; i < m*n; i++) {
        tensor[i] = rand()/(float)INT32_MAX;
    }

    return tensor;
}

BNNSNDArrayDescriptor get_desc(BNNSNDArrayFlags flags, BNNSDataLayout layout, size_t size[BNNS_MAX_TENSOR_DIMENSION], void *data) {
    BNNSNDArrayDescriptor desc;
    desc.flags = flags;
    desc.layout = layout;
    if (size[0] == 1 && size[1] != 0) {
        size[0] = size[1]; // to make rand_mat(1, n) vector initializations work
        size[1] = 0;
    } 
    list_init(desc.size, size);
    list_init(desc.stride, (size_t[]){0, 0, 0, 0, 0, 0, 0, 0 });
    desc.data = data;
    desc.data_type = BNNSDataTypeFloat32;
    desc.table_data = NULL;
    desc.table_data_type = BNNSDataTypeFloat32;
    desc.data_scale = 1.; // these fields used for converting ints to floats. doesn't hurt to leave it as 1
    desc.data_bias = 0.;

    return desc;
}

AdamOptimizer adam_optim(BNNSDataLayout layout, size_t m, size_t n) {
    BNNSNDArrayFlags flag = BNNSNDArrayFlagBackpropSet;
    float *adam_acc1_data = rand_mat(m, n);
    BNNSNDArrayDescriptor adam_acc1_desc = get_desc(flag, layout, (size_t[]){m, n, 0, 0, 0, 0, 0, 0 }, adam_acc1_data); 
    float *adam_acc2_data = rand_mat(m, n);
    BNNSNDArrayDescriptor adam_acc2_desc = get_desc(flag, layout, (size_t[]){m, n, 0, 0, 0, 0, 0, 0 }, adam_acc2_data); 
    
    // make sure to free this later

    BNNSOptimizerAdamFields adam_fields = {
        .learning_rate = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.9,
        .time_step = 1.,
        .epsilon = 1e-2,
        .gradient_scale = 1.,
        .regularization_scale = 0.01,
        .clip_gradients = true,
        .clip_gradients_min = -1000,
        .clip_gradients_max = 1000.,
    };

    return (AdamOptimizer) {    
        .adam_fields = adam_fields,
        .adam_acc1_desc = adam_acc1_desc,
        .adam_acc2_desc = adam_acc2_desc,
    };
}

DenseLayer dense_layer(size_t m, size_t n) {
    BNNSNDArrayFlags flag = BNNSNDArrayFlagBackpropSet;
    BNNSNDArrayDescriptor in_desc = get_desc(flag, BNNSDataLayoutVector, (size_t[]){m, 0, 0, 0, 0, 0, 0, 0 }, NULL);
    BNNSNDArrayDescriptor out_desc = get_desc(flag, BNNSDataLayoutVector, (size_t[]){n, 0, 0, 0, 0, 0, 0, 0 }, NULL);

    float *weights_data = rand_mat(m, n);
    BNNSNDArrayDescriptor weights_desc = get_desc(flag, BNNSDataLayoutRowMajorMatrix, (size_t[]){m, n, 0, 0, 0, 0, 0, 0 }, weights_data);

    float *bias_data = rand_mat(1, n);
    BNNSNDArrayDescriptor bias_desc = get_desc(flag, BNNSDataLayoutVector, (size_t[]){n, 0, 0, 0, 0, 0, 0, 0 }, bias_data);
    
    BNNSLayerParametersFullyConnected layer_params = {
        .i_desc = in_desc,
        .w_desc = weights_desc,
        .o_desc = out_desc,
        .bias = bias_desc,
        .activation = BNNSActivationFunctionIdentity
    };

    BNNSFilterParameters filter_params = {
        .flags = BNNSFlagsUseClientPtr,
        .n_threads = 1,
        .alloc_memory = NULL,
        .free_memory = NULL,
    };

    BNNSFilter dense_filter = BNNSFilterCreateLayerFullyConnected(&layer_params, &filter_params);

    float *in_delta_data = rand_mat(1, m);
    BNNSNDArrayDescriptor in_delta_desc = get_desc(flag, BNNSDataLayoutVector, (size_t[]){m, 0, 0, 0, 0, 0, 0, 0 }, in_delta_data); 

    float *weights_delta = rand_mat(m, n);
    BNNSNDArrayDescriptor weights_delta_desc = get_desc(flag, BNNSDataLayoutRowMajorMatrix,  (size_t[]){m, n, 0, 0, 0, 0, 0, 0 }, weights_delta); 

    float *bias_delta = rand_mat(1, n);
    BNNSNDArrayDescriptor bias_delta_desc = get_desc(flag, BNNSDataLayoutVector, (size_t[]){n, 0, 0, 0, 0, 0, 0, 0 }, bias_delta); 

    AdamOptimizer weight_optim = adam_optim(BNNSDataLayoutRowMajorMatrix, m, n);
    AdamOptimizer bias_optim = adam_optim(BNNSDataLayoutVector, 1, n);

    float *input_buffer = malloc(sizeof(float)*m);
    
    return (DenseLayer){
        .input_buffer = input_buffer,
        .weights_desc = weights_desc, 
        .bias_desc = bias_desc, 
        .in_desc = in_desc, 
        .out_desc = out_desc, 
        .in_delta_desc = in_delta_desc,
        .weights_delta_desc = weights_delta_desc, 
        .bias_delta_desc = bias_delta_desc, 
        .layer_params = layer_params, 
        .filter_params = filter_params, 
        .filter = dense_filter, 
        .weight_optim = weight_optim, 
        .bias_optim = bias_optim, 
    };
}

MSELossLayer mse_loss_layer(size_t n) {
    BNNSNDArrayFlags flag = 0;
    
    BNNSNDArrayDescriptor in_desc = get_desc(flag, BNNSDataLayoutVector, (size_t[]){n, 0, 0, 0, 0, 0, 0, 0 }, NULL);
    BNNSNDArrayDescriptor out_desc = get_desc(flag, BNNSDataLayoutVector, (size_t[]){1, 0, 0, 0, 0, 0, 0, 0 }, NULL);

    float *in_delta = rand_mat(1, n);
    float *out_delta = rand_mat(1, n);
    BNNSNDArrayDescriptor in_delta_desc = get_desc(0, BNNSDataLayoutVector, (size_t[]){n, 0, 0, 0, 0, 0, 0, 0 }, in_delta); 
    BNNSNDArrayDescriptor out_delta_desc = get_desc(0, BNNSDataLayoutVector, (size_t[]){1, 0, 0, 0, 0, 0, 0, 0 }, out_delta); 

    BNNSLayerParametersLossBase loss_params = {
        .function = BNNSLossFunctionMeanSquareError,
        .i_desc = in_desc,
        .o_desc = out_desc,
        .reduction = BNNSLossReductionSum
    };

    BNNSFilter loss_layer = BNNSFilterCreateLayerLoss(&loss_params, NULL);

    float *input_buffer = malloc(sizeof(float)*n);

    return (MSELossLayer) {
        .input_buffer = input_buffer,
        .in_desc = in_desc,
        .out_desc = out_desc,
        .in_delta_desc = in_delta_desc,
        .out_delta_desc = out_delta_desc,
        .loss_params = loss_params,
        .filter = loss_layer
    };
}

OutputLayer output_layer(size_t n) {
    return (OutputLayer) {
        .input_buffer = malloc(n*sizeof(float))
    };
}

int dense_forward(DenseLayer *dense, float *input, float *output) {
    int res = BNNSFilterApply(dense->filter, input, output);
    return res;
}

int mse_forward(MSELossLayer *mse_loss, float *target, float *result) {
    // figure out the params for a proper batch call later
    // the stride is for batch inference. how many elements to jump for the next sample
    int n = mse_loss->in_desc.size[0]; // jank way of finding stride
    int res = BNNSLossFilterApplyBatch(mse_loss->filter, 1, mse_loss->input_buffer, 
        n, target, n, NULL, 0, result, &mse_loss->in_delta_desc, n);

    return res;
}

// dense_output aka next layer's input buffer
int dense_backward(DenseLayer *dense, float *dense_output, const BNNSNDArrayDescriptor *dense_output_delta_desc) {
    // if I do batch backward, I need to set the ndarraydesc flag to 1 (accumulate)
    int m = dense->in_desc.size[0];
    int n = dense->in_desc.size[1];
    int res = BNNSFilterApplyBackwardBatch(dense->filter, 1, dense->input_buffer, m, &dense->in_delta_desc, m, dense_output, 
        n, dense_output_delta_desc, n, &dense->weights_delta_desc, &dense->bias_delta_desc);

    return res;
}

int dense_adam_optim(DenseLayer *dense) {
    BNNSNDArrayDescriptor *weights_desc_ptr = &dense->weights_desc;
    BNNSNDArrayDescriptor *bias_desc_ptr = &dense->bias_desc;
    const BNNSNDArrayDescriptor *weights_delta_desc_ptr = &dense->weights_delta_desc;
    BNNSNDArrayDescriptor *bias_delta_desc_ptr = &dense->bias_delta_desc;
    BNNSNDArrayDescriptor *adam_weight_acc_descs[2] = {&dense->weight_optim.adam_acc1_desc, &dense->weight_optim.adam_acc2_desc};
    BNNSNDArrayDescriptor *adam_bias_acc_descs[2] = {&dense->bias_optim.adam_acc1_desc, &dense->bias_optim.adam_acc2_desc};

    int res = BNNSOptimizerStep(BNNSOptimizerFunctionAdam, &dense->weight_optim.adam_fields, 1, &weights_desc_ptr, &weights_delta_desc_ptr, adam_weight_acc_descs, NULL);
    res |= BNNSOptimizerStep(BNNSOptimizerFunctionAdam, &dense->bias_optim.adam_fields, 1, &bias_desc_ptr, &bias_delta_desc_ptr, adam_bias_acc_descs, NULL);

    dense->weight_optim.adam_fields.time_step += 1;
    dense->bias_optim.adam_fields.time_step += 1;

    return res;
}