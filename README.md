# froginfer

a micro inference runtime to run llama 2 models

**heavily** inspired by karpathy's [llama2.c](https://github.com/karpathy/llama2.c), but stripped down 
aggressively to the absolute barebones, such that even a middle schooler can understand what is going on.

### files

[tokenizer.bin](https://github.com/karpathy/llama2.c/raw/refs/heads/master/tokenizer.bin): helps the model turn numbers back into text

[model.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin): llama 2 architecture model trained on tinystories dataset

## usage

git clone, run `make` at root, and then:

```bash
./llm model.bin "it was raining "
```
