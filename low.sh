
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 1 --norm 0mean,100-1000,max100,None --lr 0.00001 --device cuda:1 --dest results/LSTM/low/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 2 --norm 0mean,100-1000,max100,None --lr 0.000002 --device cuda:1 --dest results/CNN/low/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 3 --norm 0mean,100-1000,max100,None --lr 0.000009 --device cuda:1 --dest results/RCN/low/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 4 --norm 0mean,100-1000,max100,None --lr 0.000009 --device cuda:1 --dest results/rotate/low/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 5 --norm 0mean,100-1000,max100,None --lr 0.00001 --device cuda:1 --dest results/GRU/low/
