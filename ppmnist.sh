python ppmnist.py --network itcn --levels 8 --ksize 3 --nhid 96 --periodic&
python ppmnist.py --network tcn --levels 8 --ksize 7 --nhid 48 --periodic&

python ppmnist.py --network iresnet --levels 10 --ksize 7 --nhid 48 --nhid_max 200 --periodic&
python ppmnist.py --network resnet --levels 10 --ksize 7 --nhid 48 --nhid_max 200 --periodic&
