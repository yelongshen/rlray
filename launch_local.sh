python -m torch.distributed.launch --nproc_per_node=2 test_mp.py 

python -m torch.distributed.launch --nproc_per_node=8 test_mp.py 
