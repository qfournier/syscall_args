# # ! /bin/bash

# ######### Requests #########

# # N-grams
# python main.py --model ngram --order 2 --save_corpus --it 1 --max_length 256 --plot_hist
# python main.py --model ngram --order 3 --load_corpus --it 2
# python main.py --model ngram --order 4 --load_corpus --it 3

# # LSTM
# # System call name
# python main.py --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 4 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # System call name (compensated)
# python main.py --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 5 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# # System call name + timestamp
# python main.py --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 6 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # System call name + event (entry/exit + success/failure)
# python main.py --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 7 --disable_time --disable_proc --disable_pid --disable_tid
# # System call name + process (process name + pid + tid)
# python main.py --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 8 --disable_entry --disable_ret --disable_time 
# # All arguments
# python main.py --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 9

# # Transformer (LM)
# # System call name
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 10 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # System call name (compensated)
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 11 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# # System call name + timestamp
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 12 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # System call name + event (entry/exit + success/failure)
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 13 --disable_time --disable_proc --disable_pid --disable_tid
# # System call name + process (process name + pid + tid)
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 14 --disable_entry --disable_ret --disable_time 
# # All arguments
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 15

# # Transformer (MLM)
# # System call name
# python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 16 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # System call name (compensated)
# python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 17 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# # System call name + timestamp
# python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 18 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # System call name + event (entry/exit + success/failure)
# python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 19 --disable_time --disable_proc --disable_pid --disable_tid
# # System call name + process (process name + pid + tid)
# python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 20 --disable_entry --disable_ret --disable_time 
# # All arguments
# python main.py --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 21

# ######## Startup #########

# # N-grams
# python main.py --data data/startup --model ngram --order 2 --save_corpus --it 31 --max_length 256 --plot_hist
# python main.py --data data/startup --model ngram --order 3 --load_corpus --it 32
# python main.py --data data/startup --model ngram --order 4 --load_corpus --it 33

# # LSTM
# # System call name
# python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 34 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # System call name (compensated)
# python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 35 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# # System call name + timestamp
# python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 36 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # System call name + event (entry/exit + success/failure)
# python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 37 --disable_time --disable_proc --disable_pid --disable_tid
# # System call name + process (process name + pid + tid)
# python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 38 --disable_entry --disable_ret --disable_time 
# # All arguments
# python main.py --data data/startup --model lstm --lm_epochs 100 --hiddens 96 --load_corpus --it 39

# # Transformer (LM)
# # System call name
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 40 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # System call name (compensated)
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 41 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# # System call name + timestamp
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 42 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # System call name + event (entry/exit + success/failure)
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 43 --disable_time --disable_proc --disable_pid --disable_tid
# # System call name + process (process name + pid + tid)
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 44 --disable_entry --disable_ret --disable_time 
# # All arguments
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 45

# # Transformer (MLM)
# # System call name
# python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 46 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # System call name (compensated)
# python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 47 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_sys 64
# # System call name + timestamp
# python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 48 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # System call name + event (entry/exit + success/failure)
# python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 49 --disable_time --disable_proc --disable_pid --disable_tid
# # System call name + process (process name + pid + tid)
# python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 50 --disable_entry --disable_ret --disable_time 
# # All arguments
# python main.py --data data/startup --model transformer --mlm_epochs 100 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 51

# ######## Ablation Study #########

# # Request LM - no timestamp, no ordering
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 60 --disable_entry --disable_ret --disable_time --disable_order --disable_proc --disable_pid --disable_tid
# # Request LM - timestamp (8), no ordering
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 61 --disable_entry --disable_ret --disable_order --disable_proc --disable_pid --disable_tid
# # Request LM - no timestamp, ordering (8)
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 62 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # Request LM - timestamp (8), ordering (8)
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 63 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # Request LM - no timestamp, ordering (16)
# python main.py --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 64 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_order 16

# # Request LM - no timestamp, no ordering
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 65 --disable_entry --disable_ret --disable_time --disable_order --disable_proc --disable_pid --disable_tid
# # Request LM - timestamp (8), no ordering
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 66 --disable_entry --disable_ret --disable_order --disable_proc --disable_pid --disable_tid
# # Request LM - no timestamp, ordering (8)
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 67 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid
# # Request LM - timestamp (8), ordering (8)
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 68 --disable_entry --disable_ret --disable_proc --disable_pid --disable_tid
# # Request LM - no timestamp, ordering (16)
# python main.py --data data/startup --model transformer --lm_epochs 100 --layers 6 --hiddens 128 --dropout 0.1 --load_corpus --it 69 --disable_entry --disable_ret --disable_time --disable_proc --disable_pid --disable_tid --emb_order 16

######## Masking Effect #########

# Requests
python main.py --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 70 --p_mask 0.05
python main.py --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 71 --p_mask 0.1
python main.py --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 72 --p_mask 0.15
python main.py --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 73 --p_mask 0.2
python main.py --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 74 --p_mask 0.25
python main.py --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 75 --p_mask 0.30

# Ciena
python main.py --data data/startup --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 76 --p_mask 0.05
python main.py --data data/startup --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 77 --p_mask 0.1
python main.py --data data/startup --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 78 --p_mask 0.15
python main.py --data data/startup --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 79 --p_mask 0.2
python main.py --data data/startup --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 80 --p_mask 0.25
python main.py --data data/startup --model transformer --mlm_epochs 100 --lm_epochs -1 --layers 6 --hiddens 128 --batch 128 --eval 500 --dropout 0.1 --load_corpus --it 81 --p_mask 0.3
