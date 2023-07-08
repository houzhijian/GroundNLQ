# Requirements
This code requires Python, PyTorch, and a few other Python libraries. 
We recommend creating conda environment and installing all the dependencies, as follows:

- Linux
- Python 
- PyTorch 
- TensorBoard
- CUDA 
- NumPy
- Pandas
- Lmdb
- Terminaltables
- Prettytable


# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

The code should be recompiled every time you update PyTorch.
