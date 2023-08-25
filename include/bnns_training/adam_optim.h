#ifndef ANDREW_BNNS_ADAM_OPTIM
#define ANDREW_BNNS_ADAM_OPTIM

#include <vecLib/BNNS/bnns.h>

typedef struct {
    BNNSOptimizerAdamFields adam_fields;
    BNNSNDArrayDescriptor adam_acc1_desc;
    BNNSNDArrayDescriptor adam_acc2_desc;
} AdamOptimizer;

AdamOptimizer adam_optim(BNNSDataLayout layout, size_t m, size_t n);
#endif