python train.py --test --path results --network iresnet --cudnn_deterministic --min_sample 50 --hidden 16 --L 200 --kernel 5 --depth 5 --max_hidden 32 --filename macho_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network iresnet --cudnn_deterministic --min_sample 50 --hidden 16 --L 200 --kernel 3 --depth 7 --max_hidden 32 --filename asassn_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network iresnet --cudnn_deterministic --min_sample 50 --hidden 16 --L 200 --kernel 7 --depth 5 --max_hidden 32 --filename ogle3_raw.pkl --ngpu 1 --K 8 --no-log&

python train.py --test --path results --network resnet --cudnn_deterministic --min_sample 50 --hidden 16 --L 200 --kernel 7 --depth 5 --max_hidden 32 --filename macho_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network resnet --cudnn_deterministic --min_sample 50 --hidden 16 --L 200 --kernel 3 --depth 6 --max_hidden 32 --filename asassn_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network resnet --cudnn_deterministic --min_sample 50 --hidden 16 --L 200 --kernel 7 --depth 6 --max_hidden 32 --filename ogle3_raw.pkl --ngpu 1 --K 8 --no-log&

python train.py --test --path results --network itcn --min_sample 50 --L 200 --kernel 3 --depth 6 --hidden 48 --filename ogle3_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network itcn --min_sample 50 --L 200 --kernel 5 --depth 7 --hidden 24 --filename macho_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network itcn --min_sample 50 --L 200 --kernel 5 --depth 6 --hidden 48 --filename asassn_raw.pkl --ngpu 1 --K 8 --no-log&

python train.py --test --path results --network tcn --min_sample 50 --L 200 --kernel 5 --depth 7 --hidden 24 --filename ogle3_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network tcn --min_sample 50 --L 200 --kernel 5 --depth 7 --hidden 12 --filename macho_raw.pkl --ngpu 1 --K 8 --no-log&
python train.py --test --path results --network tcn --min_sample 50 --L 200 --kernel 5 --depth 6 --hidden 48 --filename asassn_raw.pkl --ngpu 1 --K 8 --no-log&

python train.py --test --path results --network gru --cudnn_deterministic --min_sample 50 --dropout 0.15 --L 200 --ngpu 1 --K 8 --depth 3 --hidden 12 --filename ogle3_raw.pkl --no-log&
python train.py --test --path results --network gru --cudnn_deterministic --min_sample 50 --dropout 0.15 --L 200 --ngpu 1 --K 8 --depth 2 --hidden 12 --filename macho_raw.pkl --no-log&
python train.py --test --path results --network gru --cudnn_deterministic --min_sample 50 --dropout 0.15 --L 200 --ngpu 1 --K 8 --depth 2 --hidden 48 --filename asassn_raw.pkl --no-log&

python train.py --test --path results --network lstm --cudnn_deterministic --min_sample 50 --dropout 0.15 --L 200 --ngpu 1 --K 8 --depth 2 --hidden 24 --filename asassn_raw.pkl --no-log&
python train.py --test --path results --network lstm --cudnn_deterministic --min_sample 50 --dropout 0.15 --L 200 --ngpu 1 --K 8 --depth 2 --hidden 48 --filename ogle3_raw.pkl --no-log&
python train.py --test --path results --network lstm --cudnn_deterministic --min_sample 50 --dropout 0.15 --L 200 --ngpu 1 --K 8 --depth 2 --hidden 12 --filename macho_raw.pkl --no-log&
