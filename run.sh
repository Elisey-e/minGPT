# 1) обучение + сохранение графика и весов
python train.py config.json

# 2) валидационная распечатка на 20 примерах
python val.py config.json

# 3) инференс (строка, пробел, символ)
python test.py "aabbaac a"     # → 4
