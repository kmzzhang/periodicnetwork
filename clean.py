import joblib
from light_curve import LightCurve
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Outlier rejection')
parser.add_argument('--file', type=str, default='data/linear/raw.pkl',
                    help='path of pkl file to be cleaned')
args = parser.parse_args()
data = joblib.load(args.file)
for lc in tqdm(data):
    lc.clean_error(thres=5, niter=3)
    lc.clean_mm(thres=4, window=3, niter=3)
    lc.fit_supersmoother()
joblib.dump(data, args.file[:-4]+'_clean.pkl')
