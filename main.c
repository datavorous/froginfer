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

int main() {
    return 0;
}