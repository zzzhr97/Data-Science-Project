
# --extract-method: sift1000, ss
# --encoding-method: bow, vlad, fisher
# --ec-n-clusters: 2, 4, 8, 16, 32, 64, 128, 256

# --max-dim: 256
# --C: 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000

extract_method="ss"
encoding_method="bow"
max_dim="256"
ec_n_clusters="2"

C="0.001 0.01 0.1 1 10 100 1000"
kernel="poly rbf"

python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 2
python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 4
python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 8
python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 16
python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 32
python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 64
python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 128
python main.py --extract-method ${extract_method} --encoding-method ${encoding_method} --max-dim ${max_dim} --C ${C} --kernel ${kernel} --ec-n-clusters 256