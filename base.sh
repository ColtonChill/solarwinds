
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 1 --norm 0mean,100-1000,max100,None --lr 0.0001 --device cuda:2 --dest results/LSTM/base/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 2 --norm 0mean,100-1000,max100,None --lr 0.00002 --device cuda:2 --dest results/CNN/base/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 3 --norm 0mean,100-1000,max100,None --lr 0.00009 --device cuda:2 --dest results/RCN/base/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 4 --norm 0mean,100-1000,max100,None --lr 0.00009 --device cuda:2 --dest results/rotate/base/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 5 --norm 0mean,100-1000,max100,None --lr 0.0001 --device cuda:2 --dest results/GRU/base/
