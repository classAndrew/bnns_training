#ifndef ANDREW_BNNS_OUTPUT_LAYER
#define ANDREW_BNNS_OUTPUT_LAYER

typedef struct {
    float *input_buffer; 
} OutputLayer;

OutputLayer output_layer(size_t n);

#endif