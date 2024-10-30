python -m torch.distributed.launch --nproc_per_node=2 test_mp.py 

python -m torch.distributed.launch --nproc_per_node=8 mini_rl_example.py 

python -m torch.distributed.launch --nproc_per_node=4 test_mp2.py 
