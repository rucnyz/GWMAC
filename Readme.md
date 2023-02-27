# GWMAC

## How to Run

Suggested parameters:

- Caltech-7

```shell
python main.py --lr 0.000001090173714489672 --lr_weight 0.11783283344491094 --loss_fn L2 --iters 2 --epochs 300 --gamma 800 --alpha 0 --data cal7 --num_workers 0 --save 1 --test_rate 0
```

- ORL

```shell
python main.py --lr 0.00012477022117309962 --lr_weight 0.08765942332555962 --loss_fn L2 --iters 6 --epochs 300 --gamma 1200 --alpha 0 --data orl --save 1 --test_rate 0
```

- Handwritten

```shell
python main.py --lr 0.00004377079516722399 --loss_fn L2 --iters 12 --epochs 300 --gamma 7600 --alpha 0 --data hw --lr_weight 0.24055548923431153 --save 1 --test_rate 0
```

- Movies

```shell
python main.py --lr 0.000059162609414369276 --loss_fn L2 --iters 14 --epochs 300 --gamma 2700 --alpha 0 --data movie --lr_weight 0.23445729088993142 --save 1 --test_rate 0
```

- 3-Sources

```shell
python main.py --lr 0.00013104678088694767 --lr_weight 0.052960546470021304 --loss_fn L2 --iters 8 --epochs 380 --gamma 100 --alpha 0 --data 3s --save 1 --test_rate 0
```

- Prokaryotic

```shell
python main.py --lr 0.00009492283128912476 --loss_fn L2 --iters 4 --epochs 300 --gamma 6900 --alpha 0 --data pro --lr_weight 0.11549647024305039 --test_rate 0 --save 1
```

## Citations

If you use this code, please cite the following paper:

**Gromov-Wasserstein Multi-Modal Alignment and Clustering**

```bibtex
@inproceedings{10.1145/3511808.3557339,
author = {Gong, Fengjiao and Nie, Yuzhou and Xu, Hongteng},
title = {Gromov-Wasserstein Multi-Modal Alignment and Clustering},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557339},
doi = {10.1145/3511808.3557339},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {603–613},
numpages = {11},
keywords = {gromov-wasserstein barycenter, kernel fusion, data alignment, multi-modal clustering, optimal transport},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```
