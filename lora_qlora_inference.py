import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from inference import LLaMA
from model import ModelArgs, Transformer

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(x, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + self.scaling * lora_out


def apply_lora(model, r=4, alpha=1.0, target_modules=["wq", "wk", "wv", "wo"]):
    for module in model.modules():
        for target in target_modules:
            if hasattr(module, target):
                orig_layer = getattr(module, target)
                if isinstance(orig_layer, nn.Linear):
                    in_features = orig_layer.in_features
                    out_features = orig_layer.out_features
                    bias = orig_layer.bias is not None
                    lora_layer = LoRALinear(in_features, out_features, r=r, alpha=alpha, bias=bias)
                    with torch.no_grad():
                        lora_layer.weight.copy_(orig_layer.weight)
                        if bias:
                            lora_layer.bias.copy_(orig_layer.bias)
                    setattr(module, target, lora_layer)
    return model


prompts = [
        "Simply put, the theory of relativity states that ",
        # # Few shot prompt
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => """
    ]

CHECKPOINTS_DIR = "Llama-2-7b/"           
TOKENIZER_PATH = "Llama-2-7b/tokenizer.model"  
MAX_SEQ_LEN = 500                        
MAX_BATCH_SIZE = len(prompts)                       
DEVICE = "cpu"                          

USE_LORA = True
LORA_RANK = 4
LORA_ALPHA = 1.0
USE_QUANTIZATION = True  


MAX_GEN_LEN = 64

def main():
    print("Loading base model...")
    model_instance = LLaMA.build(
        checkpoints_dir=CHECKPOINTS_DIR,
        tokenizer_path=TOKENIZER_PATH,
        load_model=True,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        device=DEVICE
    )
    base_model = model_instance.model

    if USE_LORA:
        print("Applying LoRA modifications...")
        base_model = apply_lora(base_model, r=LORA_RANK, alpha=LORA_ALPHA)

    model_instance.model = base_model

    print("Running inference...")
    start_time = time.time()
    tokens, output_text = model_instance.text_completion(prompts, max_gen_len=MAX_GEN_LEN)
    end_time = time.time()

    print("Generated text:")
    print(output_text[0])
    print("Inference time: {:.2f} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()
