#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int dim;
    int hidden_dim;
    int num_heads;
    int num_layers;
    int vocab_size;
    int seq_len;
    int n_kv_heads
} Config;

typedef struct {
    int *token_embedding;
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
    RunState state;
    float *data;
} Transformer;

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

static void allocState(RunState *state, Config *config) {
    int kv_dimensions = (config->dim * config->n_kv_heads) / config->num_heads;
    state->x = malloc(config->dim * sizeof(float));
    state->xb = malloc(config->dim * sizeof(float));
    state->xb2 = malloc(config->dim * sizeof(float));

    state->q = malloc(kv_dimensions * sizeof(float));
    state->k = malloc(kv_dimensions * sizeof(float));
    state->v = malloc(kv_dimensions * sizeof(float));

    state->hb = malloc(config->hidden_dim * sizeof(float));
    state->hb2 = malloc(config->hidden_dim * sizeof(float));

    state->attention = malloc(config->seq_len * sizeof(float));
    state->logits = malloc(config->vocab_size * sizeof(float));
}

static void wireWeights(Weights *weights, Config *config, float *pointer, int shared) {

    int hs = config->dim / config->num_heads;
    int kd = (config->dim * config->n_kv_heads) / config->num_heads;
    long L = config->num_layers;

    weights->token_embedding = pointer; 
    pointer += config->vocab_size * config->dim;

    weights->rms_attention = pointer;
    pointer += L* config->dim;

    weights->wQ = pointer;
    pointer += L* config->dim * config->dim;
    weights->wK = pointer;
    pointer += L * config->dim * kd;
    weights->wV = pointer;
    pointer += L * config->dim * kd;
    weights->wO = pointer;
    pointer += L * config->dim * config->dim;

    weights->rms_ffn = pointer;
    pointer += L * config->dim;

    weights->w1 = pointer;
    pointer += L * config->dim * config->hidden_dim;
    weights->w2 = pointer;
    pointer += L * config->hidden_dim * config->dim;
    weights->w3 = pointer;
    pointer += L * config->dim * config->hidden_dim;

    weights->rms_final = pointer;
    pointer += config->dim;

    pointer += config->seq_len * hs;
    pointer += config->seq_len * hs;

    weights->wcls = shared? weights->token_embedding : pointer;
}

static void loadTransformer(Transformer *transformer, const char *path) {
    FILE *file = fopen(path, "rb");
    fread(&transformer->config, sizeof(Config), 1, file);
    int shared = transformer->config.vocab_size > 0;

    transformer->config.vocab_size = abs(transformer->config.vocab_size);

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, sizeof(Config), SEEK_SET);
    long wsize = file_size - sizeof(Config);
    transformer->data = malloc(wsize);
    fread(transformer->data, 1, wsize, file);
    fclose(file);

    wireWeights(&transformer->weights, &transformer->config, transformer->data, shared);
    allocState(&transformer->state, &transformer->config);
}

static float *forward(Transformer *transformer, int *tokens, int n_tokens) {
    Config *config = &transformer->config;
    Weights *weights = &transformer->weights;
    RunState *state = &transformer->state;

    int dim = config->dim;
    int kv_dim = (dim * config->n_kv_heads) / config->num_heads;
    int head_size = dim / config->num_heads;
    int kv_mul = config->n_kv_heads / config->num_heads;
    int pos = n_tokens - 1;

    float *x = state->x;
    memcpy(x, weights->token_embedding + tokens[pos] * dim, dim * sizeof(float));

    for() {
        for() {
            // uhhh
        }
    }

    rmsnorm(x, x, weights->rms_final, dim);
    matmul(state->logits, x, weights->wcls, dim, config->vocab_size);

    return state->logits;
}

int main() {
    return 0;
}