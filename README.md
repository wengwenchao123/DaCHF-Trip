# Hyper-Trip

## Brief Introduction

This is a PyTorch implementation of **A Hypernetwork and Contrastive Learning Framework for Trip Recommendation**

## Environmental Requirements

```
numpy 
pandas
scikit-learn
pytorch
```
**Note:** Due to changes in some built-in functions across different versions of PyTorch, if an error occurs in the `dataset.py` file indicating that `torch.utils.data.dataset` has no attribute `_T_co`, you can resolve this issue by replacing `_T_co` with `T_co` in the Python file.

## Folder Structure

| Folder Name |                    Description                    |
| :---------: | :-----------------------------------------------: |
|    asset    |        Metadata and preprocessing process         |
|   results   |       Storage related experimental results        |
|     src     |            the source code of Hyper-Trip          |
|  README.md  |             This instruction document             |

## How to run this Program


```
@echo off

REM Setting Python Interpreter Path
set python_path=(alter to your python path)

python .\src\run.py --dataset Osak --lr 0.001 --batch_size 16 --d_model 32 
python .\src\run.py --dataset Glas --lr 0.001 --batch_size 16 --d_model 32 
python .\src\run.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 32 
python .\src\run.py --dataset Toro --lr 0.001 --batch_size 16 --d_model 32 
```

If your operating system is **Windows**, you can directly paste commands into the terminal to run the program just like

```
python .\src\run.py --dataset Glas --lr 0.001 --batch_size 16 --d_model 32 
```

Hope such instruction could help you with our projects.
