#ifndef ANDREW_BNNS_LOSS_LAYER
#define ANDREW_BNNS_LOSS_LAYER

#include <vecLib/BNNS/bnns.h>

typedef struct {
    float *input_buffer; 
    BNNSNDArrayDescriptor in_desc;
    BNNSNDArrayDescriptor out_desc;
    BNNSNDArrayDescriptor in_delta_desc;
    BNNSNDArrayDescriptor out_delta_desc;
    BNNSLayerParametersLossBase loss_params;
    BNNSFilter filter;
} MSELossLayer;

MSELossLayer mse_loss_layer(size_t n);
int mse_forward(MSELossLayer *mse_loss, float *target, float *result);

#endif