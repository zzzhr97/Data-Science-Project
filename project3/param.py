from texttable import Texttable
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--abs-path", type=str, default="./dataset", help="Absolute path to the datset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--extract", action="store_true", help="Extract features from the dataset.")
    parser.add_argument("--extract-method", type=str, default="sift1000", help="Local descriptor extraction method, [sift1000, ss_1, ss_2]")

    parser.add_argument("--encoding", action="store_true", help="Encode local descriptors to features.")
    parser.add_argument("--encoding-method", type=str, default="bow", choices=["bow", "vlad", "fisher"], help="Feature encoding method.")

    parser.add_argument("--kernel", nargs='+', default=["rbf"], choices=['linear', 'poly', 'rbf', 'sigmoid'], help="Kernel type for SVM.")
    parser.add_argument("--C", nargs='+', type=float, default=[1e-3], help="Regularization parameter.")
    parser.add_argument("--max-dim", type=int, default=256, help="Maximum feature dimension. If the feature dimension is larger than this value, PCA will be applied.")

    # sift
    parser.add_argument("--sift-num-points", type=int, default=1000, help="Number of SIFT points.")
    parser.add_argument("--sift-num-images", type=int, default=10, help="Number of images to show with SIFT.")
    parser.add_argument("--sift-show-images", action="store_true", help="Show images with SIFT points.")
    parser.add_argument("--sift-shuffle", action="store_true", help="Shuffle the images for SIFT.")

    # ss
    parser.add_argument("--ss-device", type=int, default=0, help="Device used for selective search, -1 for cpu.")
    parser.add_argument("--ss-scale", type=int, default=1, help="Scale parameter for selective search.")
    parser.add_argument("--ss-sigma", type=float, default=0.8, help="Sigma parameter for selective search.")
    parser.add_argument("--ss-min-size", type=int, default=50, help="Minimum size parameter for selective search.")
    parser.add_argument("--ss-num-images", type=int, default=10, help="Number of images to show with selective search.")
    parser.add_argument("--ss-image-start", type=int, default=0, help="Start index for images.")
    parser.add_argument("--ss-show-images", action="store_true", help="Show images with selective search.")

    # encoder
    parser.add_argument("--ec-n-clusters", type=int, default=64, help='Number of clusters for KMeans or GMM.')
    parser.add_argument("--ec-cotype", type=str, default='diag', help='Covariance type for GMM.')
    parser.add_argument("--ec-n-samples-per-class", type=int, default=10, help='Number of samples per class for training encoder.')
    parser.add_argument("--ec-num-samples", type=int, default=10, help="Number of samples to extract features, -1 for all.")
    parser.add_argument("--ec-verbose", type=int, default=0, help="Verbose level for KMeans or GMM.")
    
    args = parser.parse_args()
    return args

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    # Params
    - `args`: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.set_cols_dtype(['t', 't'])
    rows = [["Parameter", "Value"]] + [
        [
            k.replace("_", " ").capitalize(), 
            format_value(args[k])
        ] for k in keys
    ]
    t.add_rows(rows)
    print(t.draw())

def format_value(value):
    """
    Format the float value.
    """
    if isinstance(value, float):
        return "{:.2e}".format(value)  
    elif isinstance(value, list):
        return "..."
    else:
        return str(value)
    
if __name__ == '__main__':
    args = parse_args()
    tab_printer(args)
    print(args.abs_path)