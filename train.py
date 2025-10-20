#!/usr/bin/env python3
import os, sys, json, string, pickle
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# ---------------- Dataset ----------------

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

    def __init__(self, config, split):
        assert split in {"train", "val", "test"}
        self.config = config
        self.split = split
        self.Lmax = int(config.max_len)
        self.Wmax = len(str(self.Lmax))
        self.alphabet = ''.join(sorted(set(config.alphabet)))
        self.sep = str(config.sep_char)
        self.eos = str(config.eos_char)
        self.vocab = list(self.alphabet) + [' '] + list('0123456789') + [self.sep, self.eos]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        # max input: s(<=Lmax) + ' ' + c + SEP  => Lmax + 3
        # max output: up to Wmax digits + EOS   => Wmax + 1
        # total -1 because of autoregressive offset (cat[:-1] -> cat[1:])
        self.block_size = (self.Lmax + 3) + (self.Wmax + 1) - 1
        self._rng = torch.Generator().manual_seed(1337)

    def __len__(self):
        return 12000 if self.split == "train" else 4000

    def get_vocab_size(self): return len(self.vocab)
    def get_block_size(self): return self.block_size

    def _digits_plain(self, n: int) -> str:
        # без реверса и без паддинга; переменная длина + EOS вне цифр
        return str(n)

    def _sample_balanced_pair(self):
        # случайная длина ℓ и равномерный k∈[0..ℓ]
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

    def _sample_uniform_pair(self):
        L = int(torch.randint(1, self.Lmax + 1, (1,), generator=self._rng).item())
        idxs = torch.randint(len(self.alphabet), (L,), generator=self._rng)
        s = ''.join(self.alphabet[i] for i in idxs.tolist())
        c = self.alphabet[int(torch.randint(len(self.alphabet), (1,), generator=self._rng).item())]
        return s, c, s.count(c)

    def __getitem__(self, _):
        while True:
            s, c, k = (self._sample_balanced_pair() if self.config.balanced
                       else self._sample_uniform_pair())
            h = hash(pickle.dumps((s, c)))
            mod = h % 5
            tgt = ("test" if mod == 0 else "val" if mod == 1 else "train")
            if tgt == self.split: break

        input_str = f"{s} {c}{self.sep}"
        out_str = self._digits_plain(k) + self.eos
        cat = input_str + out_str

        # pad/truncate to fixed length (block_size+1 tokens for (x,y) shift)
        ids = [self.stoi[ch] for ch in cat]
        pad_id = self.stoi[self.eos]
        if len(ids) < self.block_size + 1:
            ids = ids + [pad_id] * (self.block_size + 1 - len(ids))
        elif len(ids) > self.block_size + 1:
            # защитное усечение (не должно срабатывать при корректном block_size)
            ids = ids[:self.block_size + 1]
        # эффективная фактическая длина исходной последовательности (после возможного усечения)
        eff_len = min(len(cat), self.block_size + 1)

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)

        # маскируем (a) входную часть, (b) правый паддинг
        y[:len(input_str)-1] = -1
        y[eff_len-1:] = -1
        return x, y

# -------------- Inference helpers --------------

@torch.no_grad()
def predict_count(model, stoi, itos, L, W, reverse_out, s, c, device):
    # ПОДПИСЬ НЕ МЕНЯЕМ, но интерпретируем: L=Wmax; reverse_out игнорируем.
    # Для совместимости передаём сюда Lmax и Wmax одинаково.
    Lmax = L
    Wmax = W
    assert len(s) <= Lmax, f"len(s)={len(s)} > max_len={Lmax}"
    # sep/eos извлекаем из словаря (они есть в stoi/itos)
    # найдём любой символ, который не цифра и не пробел, присутствующий и в stoi, и в контексте train: возьмём '|','~' по умолчанию
    sep = '|' if '|' in stoi else list(set(stoi.keys()) - set(list('0123456789') + [' '] + list(itos.values())))[0]
    eos = '~' if '~' in stoi else None
    ctx = f"{s} {c}{sep}"
    x = torch.tensor([[stoi[ch] for ch in ctx]], dtype=torch.long).to(device)
    # генерим до Wmax+1 токенов (запас для EOS), потом обрежем по EOS
    y = model.generate(x, max_new_tokens=Wmax + 1, do_sample=False)[0]
    tail = ''.join(itos[int(i)] for i in y[len(ctx):])  # только сгенерированная часть
    if eos and eos in tail:
        tail = tail[:tail.index(eos)]
    digits = ''.join(ch for ch in tail if ch.isdigit())
    try: return int(digits) if digits else 0
    except ValueError: return 0

@torch.no_grad()
def eval_loss(model, dataset, device, batches, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)
    total, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        x = x.to(device); y = y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        n += 1
        if n >= batches: break
    return total / max(1, n)

@torch.no_grad()
def eval_accuracy(model, ds: CountDataset, device, samples=1000):
    ok, tot = 0, 0
    for _ in range(samples):
        s, c, k = ds._sample_balanced_pair()
        if hash(pickle.dumps((s, c))) % 5 == 0:  # тест-подобные
            pred = predict_count(model, ds.stoi, ds.itos, ds.Lmax, ds.Wmax, False, s, c, device)
            ok += int(pred == k)
            tot += 1
    return (ok / max(1, tot)) if tot else 0.0

# ----------------- Config loader -----------------

def load_json_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_cfg(json_cfg):
    # Нельзя инициализировать CN(dict). Создаём узлы и назначаем поля явно.
    C = CN()

    # system
    C.system = CN()
    C.system.seed = json_cfg["system"]["seed"]
    C.system.work_dir = json_cfg["system"]["work_dir"]

    # data
    C.data = CN()
    C.data.max_len = json_cfg["data"]["max_len"]
    C.data.alphabet = json_cfg["data"]["alphabet"]
    C.data.reverse_out = json_cfg["data"]["reverse_out"]
    C.data.balanced = json_cfg["data"]["balanced"]
    C.data.sep_char = json_cfg["data"]["sep_char"]
    C.data.eos_char = json_cfg["data"]["eos_char"]

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = json_cfg["model"]["model_type"]

    # trainer (берём дефолт и накрываем нужными полями)
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = json_cfg["trainer"]["learning_rate"]
    C.trainer.batch_size = json_cfg["trainer"]["batch_size"]
    C.trainer.max_iters = json_cfg["trainer"]["max_iters"]
    C.trainer.num_workers = json_cfg["trainer"]["num_workers"]

    # extras (вне Trainer)
    C.extras = CN()
    C.extras.eval_interval = json_cfg["trainer"]["eval_interval"]
    C.extras.val_batches = json_cfg["trainer"]["val_batches"]
    C.extras.acc_samples = json_cfg["trainer"]["acc_samples"]

    return C

# ----------------------- Main -----------------------

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    jcfg = load_json_cfg(cfg_path)
    config = build_cfg(jcfg)

    setup_logging(config)
    set_seed(config.system.seed)
    os.makedirs(config.system.work_dir, exist_ok=True)

    train_dataset = CountDataset(config.data, split='train')
    val_dataset   = CountDataset(config.data, split='val')

    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()

    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset)

    iters, tr_losses, val_losses, val_accs = [], [], [], []

    def on_batch_end(tr):
        if tr.iter_num % 100 == 0:
            print(f"iter {tr.iter_num}: loss {tr.loss.item():.5f}")
        if tr.iter_num % config.extras.eval_interval == 0:
            model.eval()
            vl = eval_loss(model, val_dataset, tr.device, config.extras.val_batches, config.trainer.batch_size)
            va = eval_accuracy(model, val_dataset, tr.device, samples=config.extras.acc_samples)
            iters.append(tr.iter_num)
            tr_losses.append(tr.loss.item())
            val_losses.append(vl)
            val_accs.append(va)
            # чекпойнт
            ckpt = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt)
            model.train()

    trainer.set_callback('on_batch_end', on_batch_end)
    trainer.run()

    # финальный чекпойнт
    torch.save(model.state_dict(), os.path.join(config.system.work_dir, "model.pt"))

    # график
    if iters:
        fig, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(iters, tr_losses, label="train loss")
        ax1.plot(iters, val_losses, label="val loss", linestyle="--")
        ax1.set_xlabel("iter")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(iters, [a*100 for a in val_accs], label="val acc (%)", alpha=0.6)
        ax2.set_ylabel("val acc (%)")
        # легенда
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines+lines2, labels+labels2, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(config.system.work_dir, "curves.png"), dpi=150)

if __name__ == "__main__":
    main()
