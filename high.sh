
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 1 --norm 0mean,100-1000,max100,None --lr 0.001 --device cuda:3 --dest results/LSTM/high/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 2 --norm 0mean,100-1000,max100,None --lr 0.0002 --device cuda:3 --dest results/CNN/high/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 3 --norm 0mean,100-1000,max100,None --lr 0.0009 --device cuda:3 --dest results/RCN/high/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 4 --norm 0mean,100-1000,max100,None --lr 0.0009 --device cuda:3 --dest results/rotate/high/
python3 caller.py --epochs 700 --lam 0.0,0.3,0.003,0.0001 --netNumber 5 --norm 0mean,100-1000,max100,None --lr 0.001 --device cuda:3 --dest results/GRU/high/
