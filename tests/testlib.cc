#include <gtest/gtest.h>

extern "C" {
    #include <bnns_training/adam_optim.h>
    #include <bnns_training/dense_layer.h>
    #include <bnns_training/loss_layer.h>
    #include <bnns_training/output_layer.h>
}

TEST(TestTest, Test1) {
    float input[] = {1, 2, 3, 4};
    float labels[] = {1, 3, 5};

    uint32_t m = 4;
    uint32_t n = 3;

    DenseLayer dense1 = dense_layer(m, 8); 
    DenseLayer dense2 = dense_layer(8, n); 
    MSELossLayer mse_loss = mse_loss_layer(n);
    OutputLayer last_layer = output_layer(1);

    for (int iter = 0; iter < 100; iter++) {
        // printf("\n\niter: %d\n", iter);
        // since dense1 is the first layer, initialize its input buffer w/ sample
        memcpy(dense1.input_buffer, input, sizeof(input));
        int res = dense_forward(&dense1, dense1.input_buffer, dense2.input_buffer);
        res = dense_forward(&dense2, dense2.input_buffer, mse_loss.input_buffer);
        res = mse_forward(&mse_loss, labels, last_layer.input_buffer);

        printf("mse loss: %f\n", *last_layer.input_buffer);

        res = dense_backward(&dense2, mse_loss.input_buffer, &mse_loss.in_delta_desc);
        res = dense_adam_optim(&dense2);
        res = dense_backward(&dense1, dense2.input_buffer, &dense2.in_delta_desc);
        res = dense_adam_optim(&dense1);
    }

    for (int i =0; i < m*n; i++) {
        printf("%f ", ((float*)dense1.weights_desc.data)[i]);
    }
    puts(" ");
    for (int i =0; i < n; i++) {
        printf("%f ", ((float*)dense1.bias_desc.data)[i]);
    }
    puts(" ");

    BNNSFilterDestroy(dense1.filter);
}