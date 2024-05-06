
# --extract-method: sift1000, ss
# --encoding-method: bow, vlad, fisher
# --ec-n-clusters
# --ec-cotype
# --ec-n-samples-per-class
# --ec-num-samples
# --ec-verbose

extract_method="ss"
encoding_method="fisher"
ec_n_samples_per_class="25"

python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 2 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1
python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 4 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1
python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 8 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1
python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 16 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1
python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 32 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1
python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 64 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1
python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 128 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1
python encoder.py --extract-method ${extract_method} --encoding-method ${encoding_method} \
    --ec-n-clusters 256 --ec-cotype diag --ec-n-samples-per-class ${ec_n_samples_per_class} \
    --ec-num-samples -1 --ec-verbose 1