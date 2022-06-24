# mgap
The code for "Masked Graph Auto-Encoder Constrained Graph Pooling" (ECML-PKDD 2022)



# Requirements


```

conda create -c conda-forge -n my-rdkit-env rdkit python=3.7

pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html

pip install transformers

```



# Run Code

```python

python main.py --model SAG --dataset DD --loss_alpha 0 --loss_beta 0

```
