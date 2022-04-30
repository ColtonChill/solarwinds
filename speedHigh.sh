
python3 caller.py --epochs 1000 --lam 0.0 --netNumber 1 --norm 0mean,100-1000,max100,None --lr 0.001 --device cuda:3 --dest results/LSTM/low --targets 21
python3 caller.py --epochs 1000 --lam 0.0 --netNumber 2 --norm 0mean,100-1000,max100,None --lr 0.0002 --device cuda:3 --dest results/CNN/low --targets 21
python3 caller.py --epochs 1000 --lam 0.0 --netNumber 3 --norm 0mean,100-1000,max100,None --lr 0.0009 --device cuda:3 --dest results/RCN/low --targets 21
python3 caller.py --epochs 1000 --lam 0.0 --netNumber 4 --norm 0mean,100-1000,max100,None --lr 0.0009 --device cuda:3 --dest results/rotate/low --targets 21
python3 caller.py --epochs 1000 --lam 0.0 --netNumber 5 --norm 0mean,100-1000,max100,None --lr 0.001 --device cuda:3 --dest results/GRU/low --targets 21
