
#define syn(weights, in, neuron, nth) \
    weights[(neuron)*(1+(in)*2) + (nth)]

void calc(uint in, uint out, __global float *weights, float *input, float *output) {
    for (uint i = 0; i < out; i++) {
        output[i] = syn(weights, in, i, 0);
        for (uint j = 0; j < in; j++) {
            float a = syn(weights, in, i, 1+j);
            float b = syn(weights, in, i, 1+in+j);
            output[i]+= b / (1.f + exp(input[j]*a));
        }
    }
}

__kernel void process(
                  uint layer_c,
                  __global uint *net,
                  __global float *weights,
                  __global float *data,
                  __global float *labels,
                  __global float *fit
                  ) {

    uint gid = get_global_id(0);

    float tmp_in[1024];
    float tmp_out[1024];

    // float *tmp_in; tmp_in = tmp_a;
    // float *tmp_out; tmp_out = tmp_b;

    __global float *my_data = &data[gid*net[0]];
    __global float *my_labels = &labels[gid*net[layer_c-1]];

    // Copy inputs to start the machine
    for (int i = 0; i < net[0]; i++) {
        tmp_in[i] = my_data[i];
    }


    // Feed forward
    for (int i = 1; i < layer_c; i++) {
        // float *c = NULL;
        // c = tmp_in;
        // tmp_in = tmp_out;
        // tmp_out = c;

        calc(net[i-1], net[i], weights, tmp_in, tmp_out);
        weights+= (net[i-1]*2+1)*net[i];

        for (int j = 0; j < net[i]; j++) {
            tmp_in[j] = tmp_out[j];
        }
    }

    // Calculate fitness
    float fitness = 0;



    int M = 0;
    for (int i = 1; i < net[layer_c-1]; i++) {
        if (tmp_out[i] > tmp_out[M]) M = i;
    }
    fitness+= my_labels[M];


    for (int i = 0; i < net[layer_c-1]; i++) {
        // if (tmp_out[i] < 0) tmp_out[i] = 0;
        // else if (tmp_out[i] > 1) tmp_out[i] = 1;
        if (tmp_out[i] < 0) tmp_out[i] = 0;
        if (tmp_out[i] > 1) tmp_out[i] = 1;
        float err = tmp_out[i] - my_labels[i];
        if (err < 0) err*= -1;
        // float err = 0 - my_labels[i];
        fitness-= (err)*(err);
        // fitness-= err;
    }
    fit[gid] = fitness/net[layer_c-1];

}
