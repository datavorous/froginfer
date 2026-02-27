#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int dim;
    int hidden_dim;
    int num_heads;
    int num_layers;
    int vocab_size;
    int seq_len;
    int n_kv_heads;
} Config;

typedef struct {
    int *token_embeddings;
    float *rms_attention;
    float *wQ, *wK, *wV, *wO;
    float *w1, *w2, *w3;
    float *rms_ffn;
    float *rms_final;
    float *wcls;
} Weights;

typedef struct {
    float *x; 
    float *xb, *xb2;
    float *q, *k, *v;
    float *hb, *hb2;
    float *attention;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState run_state;
    float *data;
} Transformer;

typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_len;
    unsigned char byte_pieces[512];
} Tokenizer;


static void matmul(float *out, float *x, float *W, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0;
        for(int j =0; j < n; j++)
            val += x[j] * W[j * d + i];
        out[i] = val;
    }
}

static void rmsnorm(float *out, float *x, float *w, int n) {
    float ss = 0;
    for(int i = 0; i < n; i++)
        ss += x[i] * x[i];
    ss = 1/sqrtf(ss/n + 1e-6);
    for(int i = 0; i < n; i++)
        out[i] = x[i] * ss * w[i];
}

static void softmax(float *x, int n) {
    float max = x[0];
    for(int i = 1; i < n; i++)
        if(x[i] > max) max = x[i];
    float sum = 0;
    for(int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for(int i = 0; i < n; i++)
        x[i] /= sum;
}


int main() {
    return 0;
}