# ! /bin/bash

######### Requests #########

# N-grams
python main.py --model ngram --order 2 --save_corpus --it 1 --max_length 256 --plot_hist
python main.py --model ngram --order 3 --load_corpus --it 2
python main.py --model ngram --order 4 --load_corpus --it 3

# LSTM
# System call name
python main.py --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 4 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# System call name (compensated)
python main.py --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 5 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# System call name + timestamp
python main.py --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 6 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# System call name + event (entry/exit + success/failure)
python main.py --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 7 --disable_time --disable_proc --disable_pid --disable_tid
# System call name + process (process name + pid + tid)
python main.py --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 8 --disable_entry --disable_ret --disable_time 
# All arguments
python main.py --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 9

# Transformer (LM)
# System call name
python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --early_stopping 10 --dropout 0 --load_corpus --it 10 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# System call name (compensated)
python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --early_stopping 10 --dropout 0 --load_corpus --it 11 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# System call name + timestamp
python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --early_stopping 10 --dropout 0 --load_corpus --it 12 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# System call name + event (entry/exit + success/failure)
python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --early_stopping 10 --dropout 0 --load_corpus --it 13 --disable_time --disable_proc --disable_pid --disable_tid
# System call name + process (process name + pid + tid)
python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --early_stopping 10 --dropout 0 --load_corpus --it 14 --disable_entry --disable_ret --disable_time 
# All arguments
python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --early_stopping 10 --dropout 0 --load_corpus --it 15

# Transformer (MLM)
# System call name
python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 16 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# System call name (compensated)
python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 17 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# System call name + timestamp
python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 18 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# System call name + event (entry/exit + success/failure)
python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 19 --disable_time --disable_proc --disable_pid --disable_tid
# System call name + process (process name + pid + tid)
python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 20 --disable_entry --disable_ret --disable_time 
# All arguments
python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 21

# MLM different percentage of selected input
python main.py --model transformer --p_mask 0.05 --mlm_epochs 100 --dropout 0.1 --layers 6 --hiddens 128 --batch 128 --eval 500 --load_corpus --it 100 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --model transformer --p_mask 0.10 --mlm_epochs 100 --dropout 0.1 --layers 6 --hiddens 128 --batch 128 --eval 500 --load_corpus --it 101 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --model transformer --p_mask 0.15 --mlm_epochs 100 --dropout 0.1 --layers 6 --hiddens 128 --batch 128 --eval 500 --load_corpus --it 102 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --model transformer --p_mask 0.20 --mlm_epochs 100 --dropout 0.1 --layers 6 --hiddens 128 --batch 128 --eval 500 --load_corpus --it 103 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --model transformer --p_mask 0.25 --mlm_epochs 100 --dropout 0.1 --layers 6 --hiddens 128 --batch 128 --eval 500 --load_corpus --it 104 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid

######## Startup #########

# N-grams
python main.py --data data/startup --model ngram --order 2 --save_corpus --it 31 --max_length 256 --plot_hist
python main.py --data data/startup --model ngram --order 3 --load_corpus --it 32
python main.py --data data/startup --model ngram --order 4 --load_corpus --it 33

# LSTM
# System call name
python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 34 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# System call name (compensated)
python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 35 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# System call name + timestamp
python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 36 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# System call name + event (entry/exit + success/failure)
python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 37 --disable_time --disable_proc --disable_pid --disable_tid
# System call name + process (process name + pid + tid)
python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 38 --disable_entry --disable_ret --disable_time 
# All arguments
python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --early_stopping 2 --load_corpus --it 39

# Transformer (LM)
# System call name
python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --dropout 0 --load_corpus --it 40 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# System call name (compensated)
python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --dropout 0 --load_corpus --it 41 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# System call name + timestamp
python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --dropout 0 --load_corpus --it 42 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# System call name + event (entry/exit + success/failure)
python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --dropout 0 --load_corpus --it 43 --disable_time --disable_proc --disable_pid --disable_tid
# System call name + process (process name + pid + tid)
python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --dropout 0 --load_corpus --it 44 --disable_entry --disable_ret --disable_time 
# All arguments
python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --batch 64 --dropout 0 --load_corpus --it 45

# Transformer (MLM)
# System call name
python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 46 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# System call name (compensated)
python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 47 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# System call name + timestamp
python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 48 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# System call name + event (entry/exit + success/failure)
python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 49 --disable_time --disable_proc --disable_pid --disable_tid
# System call name + process (process name + pid + tid)
python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 50 --disable_entry --disable_ret --disable_time 
# All arguments
python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 51

# MLM different percentage of selected input
python main.py --data data/startup --p_mask 0.05 --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 105 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --data data/startup --p_mask 0.10 --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 106 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --data data/startup --p_mask 0.15 --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 107 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --data data/startup --p_mask 0.20 --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 108 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
python main.py --data data/startup --p_mask 0.25 --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 109 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
