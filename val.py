#!/usr/bin/env python3
import os, sys, json, string, pickle
import torch
from torch.utils.data import Dataset
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

# те же компоненты, что и в train.py (коротко)
class CountDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.max_len = 16
        C.alphabet = string.ascii_lowercase
        C.reverse_out = False
        C.balanced = True
        C.sep_char = "|"
        C.eos_char = "~"
        return C
    def __init__(self, config):
        self.config = config
        self.Lmax = int(config.max_len)
        self.Wmax = len(str(self.Lmax))
        self.alphabet = ''.join(sorted(set(config.alphabet)))
        self.sep = str(config.sep_char)
        self.eos = str(config.eos_char)
        self.vocab = list(self.alphabet) + [' '] + list('0123456789') + [self.sep, self.eos]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self._rng = torch.Generator().manual_seed(1337)
    def _sample_balanced_pair(self):
        L = int(torch.randint(1, self.Lmax + 1, (1,), generator=self._rng).item())
        k = int(torch.randint(0, L + 1, (1,), generator=self._rng).item())
        c = self.alphabet[int(torch.randint(0, len(self.alphabet), (1,), generator=self._rng).item())]
        pos = torch.randperm(L, generator=self._rng)[:k].tolist()
        alph_wo_c = [ch for ch in self.alphabet if ch != c] or [c]
        base_idxs = torch.randint(0, len(alph_wo_c), (L,), generator=self._rng).tolist()
        s_list = [alph_wo_c[i] for i in base_idxs]
        for i in pos: s_list[i] = c
        s = ''.join(s_list)
        return s, c, k

@torch.no_grad()
def predict_count(model, stoi, itos, Lmax, Wmax, s, c, sep, eos, device):
    assert len(s) <= Lmax
    ctx = f"{s} {c}{sep}"
    x = torch.tensor([[stoi[ch] for ch in ctx]], dtype=torch.long).to(device)
    y = model.generate(x, max_new_tokens=Wmax + 1, do_sample=False)[0]
    tail = ''.join(itos[int(i)] for i in y[len(ctx):])
    if eos in tail:
        tail = tail[:tail.index(eos)]
    digits = ''.join(ch for ch in tail if ch.isdigit())
    try:
        return int(digits) if digits else 0
    except ValueError:
        return 0

def load_json_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    jcfg = load_json_cfg(cfg_path)

    data_cfg = CN()
    data_cfg.max_len    = jcfg["data"]["max_len"]
    data_cfg.alphabet   = jcfg["data"]["alphabet"]
    data_cfg.reverse_out= False
    data_cfg.balanced   = True
    data_cfg.sep_char   = jcfg["data"]["sep_char"]
    data_cfg.eos_char   = jcfg["data"]["eos_char"]

    work_dir = jcfg["system"]["work_dir"]
    model_type = jcfg["model"]["model_type"]
    ckpt = os.path.join(work_dir, "model.pt")

    ds = CountDataset(data_cfg)

    mcfg = GPT.get_default_config()
    mcfg.model_type = model_type
    mcfg.vocab_size = len(ds.vocab)
    mcfg.block_size = (ds.Lmax + ds.Wmax + 3)
    model = GPT(mcfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    print("20 random samples:")
    correct = 0
    for _ in range(20):
        s, c, k = ds._sample_balanced_pair()
        pred = predict_count(model, ds.stoi, ds.itos, ds.Lmax, ds.Wmax, s, c, ds.sep, ds.eos, device)
        ok = (pred == k)
        correct += int(ok)
        print(f"{s} {c} -> {pred}  (gt={k}) {'✓' if ok else '✗'}")
    print(f"Accuracy: {correct}/20 = {100.0*correct/20:.1f}%")

if __name__ == "__main__":
    main()
