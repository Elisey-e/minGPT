#!/usr/bin/env python3
import os, sys, json, string
import torch
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

class CountTokenizer:
    def __init__(self, alphabet, Lmax, sep_char, eos_char):
        self.L = Lmax
        self.alphabet = ''.join(sorted(set(alphabet)))
        self.sep = str(sep_char)
        self.eos = str(eos_char)
        self.vocab = list(self.alphabet) + [' '] + list('0123456789') + [self.sep, self.eos]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

@torch.no_grad()
def predict_count(model, tok: CountTokenizer, s, c, device):
    assert len(s) <= tok.L, f"len(s)={len(s)} > max_len={tok.L}"
    Wmax = len(str(tok.L))
    ctx = f"{s} {c}{tok.sep}"
    x = torch.tensor([[tok.stoi[ch] for ch in ctx]], dtype=torch.long).to(device)
    y = model.generate(x, max_new_tokens=Wmax + 1, do_sample=False)[0]
    tail = ''.join(tok.itos[int(i)] for i in y[len(ctx):])
    if tok.eos in tail:
        tail = tail[:tail.index(tok.eos)]
    digits = ''.join(ch for ch in tail if ch.isdigit())
    try: return int(digits) if digits else 0
    except ValueError: return 0

def load_json_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py \"<string> <char>\"")
        sys.exit(1)
    inp = sys.argv[1].rstrip("\n")
    if " " not in inp:
        print("Input must be: <string_without_spaces><space><one_char>")
        sys.exit(1)
    s, c = inp.rsplit(" ", 1)
    if len(c) != 1 or " " in s:
        print("Input must be: <string_without_spaces><space><one_char>")
        sys.exit(1)

    cfg_path = "config.json"
    jcfg = load_json_cfg(cfg_path)
    L = int(jcfg["data"]["max_len"])
    alphabet = jcfg["data"]["alphabet"]
    sep_char = jcfg["data"]["sep_char"]
    eos_char = jcfg["data"]["eos_char"]
    model_type = jcfg["model"]["model_type"]
    work_dir = jcfg["system"]["work_dir"]
    ckpt = os.path.join(work_dir, "model.pt")

    tok = CountTokenizer(alphabet, L, sep_char, eos_char)
    mcfg = GPT.get_default_config()
    mcfg.model_type = model_type
    mcfg.vocab_size = len(tok.vocab)
    mcfg.block_size = (L + len(str(L)) + 3)
    model = GPT(mcfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    print(predict_count(model, tok, s, c, device))

if __name__ == "__main__":
    main()
