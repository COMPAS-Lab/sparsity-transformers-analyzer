

for %%a in (2 3 4 5 6 7 8 16) do (
    python -W ignore roberta_squad_analyzer.py -e -aq %%a.0
    mkdir quantized_params\roberta-squad-quant-uniform-slog-mean\%%a
    move params\*_all.npy quantized_params\roberta-squad-quant-uniform-slog-mean\%%a\.
)