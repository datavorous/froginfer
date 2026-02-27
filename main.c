#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    int dim;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float *token_embedding;
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

typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_len;
} Tokenizer;

static void matmul(float *out, float *x, float *W, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0;
        for(int j =0; j < n; j++)
            val += x[j] * W[i * n + j];
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

    state->q = malloc(config->dim * sizeof(float));
    state->k = malloc(kv_dimensions * sizeof(float));
    state->v = malloc(kv_dimensions * sizeof(float));

    state->hb = malloc(config->hidden_dim * sizeof(float));
    state->hb2 = malloc(config->hidden_dim * sizeof(float));

    state->attention = malloc(config->num_heads * config->seq_len * sizeof(float));
    state->logits = malloc(config->vocab_size * sizeof(float));
    state->key_cache = calloc(config->num_layers * config->seq_len * kv_dimensions, sizeof(float));
    state->value_cache = calloc(config->num_layers * config->seq_len * kv_dimensions, sizeof(float));
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

static void loadTokenizer(Tokenizer *t, const char *path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab  = malloc(vocab_size * sizeof(char *));
    t->scores = malloc(vocab_size * sizeof(float));

    FILE *f = fopen(path, "rb");
    fread(&t->max_token_len, sizeof(int), 1, f);
    for (int i = 0; i < vocab_size; i++) {
        int len;
        fread(&t->scores[i], sizeof(float), 1, f);
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = malloc(len + 1);
        fread(t->vocab[i], len, 1, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

static int vocabLookup(Tokenizer *t, const char *str) {
    for (int i = 0; i < t->vocab_size; i++)
        if (strcmp(t->vocab[i], str) == 0) return i;
    return -1;
}

static int encode(Tokenizer *t, const char *text, int *tokens) {
    int n = 0;
    char buf[512];

    tokens[n] = 1;
    n++;

    for (const char *c = text; *c; c++) {
        buf[0] = *c; buf[1] = '\0';
        int id = vocabLookup(t, buf);
        tokens[n++] = (id != -1) ? id : (unsigned char)*c + 3;
    }

    while (1) {
        float best = -1e10f;
        int best_id = -1, best_pos = -1;
        for (int i = 0; i < n - 1; i++) {
            snprintf(buf, sizeof(buf), "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = vocabLookup(t, buf);
            if (id != -1 && t->scores[id] > best) {
                best = t->scores[id]; best_id = id; best_pos = i;
            }
        }
        if (best_pos == -1) break;
        tokens[best_pos] = best_id;
        for (int i = best_pos + 1; i < n - 1; i++) tokens[i] = tokens[i+1];
        n--;
    }
    return n;
}

static const char *decode(Tokenizer *t, int prev, int cur) {
    const char *s = t->vocab[cur];
    if (prev == 1 && s[0] == ' ') s++;
    unsigned char byte_val;
    if (sscanf(s, "<0x%02hhX>", &byte_val) == 1) {
        static char byte_piece[2];
        byte_piece[0] = (char)byte_val;
        byte_piece[1] = '\0';
        s = byte_piece;
    }
    return s;
}


static float *forward(Transformer *transformer, int token, int pos) {
    Config *config = &transformer->config;
    Weights *weights = &transformer->weights;
    RunState *state = &transformer->state;

    int dim = config->dim;
    int kv_dim = (dim * config->n_kv_heads) / config->num_heads;
    int head_size = dim / config->num_heads;
    int kv_mul = config->num_heads / config->n_kv_heads;

    float *x = state->x;
    memcpy(x, weights->token_embedding + token * dim, dim * sizeof(float));

    for (int l = 0; l < config->num_layers; l++) {

        rmsnorm(state->xb, x, weights->rms_attention + l * dim, dim);
        matmul(state->q, state->xb, weights->wQ + l * dim * dim, dim, dim);

        for (int i = 0; i < dim; i += 2) {
            float freq  = 1 / powf(10000, (i % head_size) / (float)head_size);
            float angle = pos * freq;
            float c_ = cosf(angle), s_ = sinf(angle);
            float v0 = state->q[i], v1 = state->q[i+1];
            state->q[i]   = v0 * c_ - v1 * s_;
            state->q[i+1] = v0 * s_ + v1 * c_;
        }

        matmul(state->k, state->xb, weights->wK + l * dim * kv_dim, dim, kv_dim);
        matmul(state->v, state->xb, weights->wV + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < kv_dim; i += 2) {
            float freq  = 1.0f / powf(10000.0f, (i % head_size) / (float)head_size);
            float angle = pos * freq;
            float c_ = cosf(angle), s_ = sinf(angle);
            float v0 = state->k[i], v1 = state->k[i + 1];
            state->k[i] = v0 * c_ - v1 * s_;
            state->k[i + 1] = v0 * s_ + v1 * c_;
        }

        int loff = l * config->seq_len * kv_dim;
        memcpy(state->key_cache + loff + pos * kv_dim, state->k, kv_dim * sizeof(float));
        memcpy(state->value_cache + loff + pos * kv_dim, state->v, kv_dim * sizeof(float));

        for (int h = 0; h < config->num_heads; h++) {
            float *q_h = state->q + h * head_size;
            float *att_h = state->attention + h * config->seq_len;
            float *xb_h = state->xb + h * head_size;

            for (int t = 0; t <= pos; t++) {
                float *k_h = state->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q_h[i] * k_h[i];
                att_h[t] = score / sqrtf(head_size);
            }

            softmax(att_h, pos + 1);

            memset(xb_h, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v_h = state->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att_h[t];
                for (int i = 0; i < head_size; i++) xb_h[i] += a * v_h[i];
            }
        }

        matmul(state->xb2, state->xb, weights->wO + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += state->xb2[i];

        rmsnorm(state->xb, x, weights->rms_ffn + l * dim, dim);
        matmul(state->hb,  state->xb, weights->w1 + l * dim * config->hidden_dim, dim, config->hidden_dim);
        matmul(state->hb2, state->xb, weights->w3 + l * dim * config->hidden_dim, dim, config->hidden_dim);
        for (int i = 0; i < config->hidden_dim; i++) {
            float v = state->hb[i];
            state->hb[i] = v / (1 + expf(-v)) * state->hb2[i];
        }
        matmul(state->xb, state->hb, weights->w2 + l * config->hidden_dim * dim, config->hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += state->xb[i];
    }

    rmsnorm(x, x, weights->rms_final, dim);
    matmul(state->logits, x, weights->wcls, dim, config->vocab_size);

    return state->logits;
}

int main(int argc, char *argv[]) {

    Transformer transformer = {0};
    Tokenizer tokenizer = {0};

    loadTransformer(&transformer, argv[1]);
    loadTokenizer(&tokenizer, argv[2], transformer.config.vocab_size);

    int tokens[512];
    int n = encode(&tokenizer, argv[3], tokens);
    int steps = (argc >= 5) ? atoi(argv[4]) : 800;
    if (steps > transformer.config.seq_len) steps = transformer.config.seq_len;
    if (n > transformer.config.seq_len) n = transformer.config.seq_len;
    if (steps < n) steps = n;

    int token = tokens[0];
    int prev = 1;
    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(&transformer, token, pos);

        int next = 0;
        for (int i = 1; i < transformer.config.vocab_size; i++)
            if (logits[i] > logits[next]) next = i;

        if (pos < n - 1) {
            prev = token;
            token = tokens[pos + 1];
        } else {
            if (next == 1) break;
            printf("%s", decode(&tokenizer, prev, next));
            fflush(stdout);
            prev = next;
            token = next;
        }
    }
    printf("\n");
    return 0;
}
