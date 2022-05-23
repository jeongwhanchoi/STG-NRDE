# Graph Neural Rough Differential Equations for Traffic Forecasting

##
## Setup Python environment for STG-NRDE
Install python environment
```{bash}
$ conda env create -f environment.yml 
```


## Reproducibility
#### In terminal
- Run the shell file (at the root of the project)

```{bash}
$ bash run_pemsd4.sh
```

- Run the python file (at the `model` folder)
```{bash}
$ cd model

$ python Run.py --dataset='PEMSD4' --model='GRDE' --model_type='rde2' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=2 --lr_init=0.001 --weight_decay=1e-3 --epochs=200 --comment="" --input_dim=3 --depth=2 --wnd_len=2 --device=0 --tensorboard
```

### Usage
#### PeMSD3
```{bash}
python Run.py --dataset='PEMSD3' --model='GRDE' --model_type='rde2' --embed_dim=5 --hid_dim=64 --hid_hid_dim=64 --num_layers=1 --lr_init=0.001 --weight_decay=2e-3 --epochs=200 --tensorboard --comment="" --input_dim=5 --depth=3 --wnd_len=2 --device=0
```

#### PeMSD4
```{bash}
python Run.py --dataset='PEMSD4' --model='GRDE' --model_type='rde2' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=2 --lr_init=0.001 --weight_decay=1e-3 --epochs=200 --comment="" --input_dim=3 --depth=2 --wnd_len=2 --device=0 --tensorboard
```

#### PeMSD7
```{bash}
python Run.py --dataset='PEMSD7' --model='GRDE' --model_type='rde2' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=1 --lr_init=0.001 --weight_decay=8e-4 --epochs=200 --comment="" --input_dim=3 --depth=2 --wnd_len=2 --device=0 --tensorboard
```

#### PeMSD8
```{bash}
python Run.py --dataset='PEMSD8' --model='GRDE' --model_type='rde2' --embed_dim=3 --hid_dim=32 --hid_hid_dim=256 --num_layers=1 --lr_init=8e-4 --weight_decay=2e-3 --epochs=200 --comment="" --input_dim=3 --depth=2 --wnd_len=2 --device=0 --tensorboard
```

#### PeMSD7(M)
```
python -u Run_cde.py --dataset='PEMSD7M' --model='GRDE' --model_type='rde2' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=2 --lr_init=0.001 --weight_decay=1e-3 --epochs=200 --comment="" --input_dim=3 --depth=2 --wnd_len=2 --device=0 --tensorboard
```

#### PeMSD7(L)
```
nohup python -u Run_cde.py --dataset='PEMSD7L' --model='GRDE' --model_type='rde2' --embed_dim=10 --hid_dim=32 --hid_hid_dim=64 --num_layers=2 --lr_init=0.001 --weight_decay=1e-3 --epochs=200 --comment="" --input_dim=8 --depth=4 --wnd_len=2 --device=0 --tensorboard
```