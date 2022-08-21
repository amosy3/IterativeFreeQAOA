## Iterative-Free Quantum Approximate Optimization Algorithm Using Neural Networks
This is an official implementation of ***Iterative-Free Quantum Approximate Optimization Algorithm Using Neural Networks*** paper. [[Link]](https://amosy3.github.io/papers/QAOA_init.pdf)

<center>
<img width="500" src="QAOA_init.png">
</center>
  

#### Installation
- Run: ```pip install -r requirements.txt```


#### Create train dataset
- Run: ```python maxcut_generator.py --random_prob --number_of_nodes=14 --number_of_layers=2```

#### Create test dataset
- Run: ```python maxcut_untrained_generator.py --random_prob --number_of_nodes=14 --number_of_layers=2```

#### Train neural network
- Run: ```python train_network.py --train_csv=<train_file_path> --test_csv=<test_file_path> --number_of_nodes=14```


#### Citation

If you find this repository to be useful in your own research, please consider citing the following paper:

```bib
@article{TBD
}
```
