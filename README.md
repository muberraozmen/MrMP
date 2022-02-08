# MrMP
This repository contains a PyTorch implementation of our ICASSP 2022 paper [Multi-relation Message Passing for Multi-label Text Classification](https://arxiv.org).

<img align="center"  src="/figures/model.pdf" alt="...">

#### Requirements
- Python 3.6
- PyTorch 1.10
- Numpy 1.19.1
- tqdm 4.62.3

#### Usage
1. install the required packages and their dependencies, if your environment does not have them already
   ```
   pip install -r requirements.txt
   ```
   
2. download and/or prepare data
    - if you would like to use benchmark datasets in the paper please download [here]()
    - if you would like to use your own dataset bring it into following format: 
   ```
   data = {'train': {'src': List[List[int]], 'tgt': List[List[int]]},
           'valid': {'src': List[List[int]], 'tgt': List[List[int]]},
           'test' : {'src': List[List[int]], 'tgt': List[List[int]]}, 
           'dict' : {'src': Dict[int, str], 'tgt': Dict[int, str]}
           }
   ```

3. pass parameters' settings optional in `main.py` according to your needs, note that the defaults of hyperparameters are set to tuned values according to the paper, e.g.
   ```
   dataset=bibtex
   name=mrmp
   python3.8 -u main.py -dataset $dataset -name $name -mrmp_on $true
   ```
4. run `python main.py -configuration config.json`

#### Citation
```bib
@inproceedings{MrMP_Ozmen22,
	author       = {Ozmen, M. and Zhang, H. and Wang, P. and Coates, M.},
	title        = {Multi-relation Message Passing for Multi-label Text Classification},
	booktitle    = {Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)},
	month = "May",
	year = "2022",
	}
```
The implementation is mainly adapted from [[Transformer]](https://github.com/jadore801120/attention-is-all-you-need-pytorch).
For any questions or comments please start an issue or contact [Muberra](http://muberraozmen.github.io).