# Language Models Deep Dive

By: [Vineet Kumar Singh](https://www.twitter.com/viiitdmj)

AWD-LSTM and Neural Cache are recent sota in language modelling task. The [paper](https://github.com/salesforce/awd-lstm-lm) seems to be written well, and has a bunch of cool tricks. We will understand each of those tricks in detal and then move onto 
[An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/pdf/1803.08240.pdf) and [Quasi-Recurrent Neural Networks: QRNN's](https://arxiv.org/pdf/1611.01576.pdf).

```Architecture```: Three-layered LSTM model with 1150 units in the hidden state and an embedding size of 400.

- ```Attention Material```
    - [Colin Raffel - Doing Strange Things with Attention](https://www.youtube.com/watch?v=YtHjmm9Cx3s)
    - [Attention and Memory in Deep Learning Networks](https://www.youtube.com/watch?v=uuPZFWJ-4bE&t=1261s9)
    - [Original Atention Paper](https://arxiv.org/abs/1409.0473)
    - [Google's NMT](https://arxiv.org/abs/1609.08144v2)

## Neural language model regularization techniques 

### 1. Weight-dropped LSTM (DropConnect)
  - [Regularization of Neural Networks using DropConnect](https://cs.nyu.edu/~wanli/dropc/dropc.pdf)
  - Instead of setting randomly selected subset of activations to zero, set a randomly selected subset of weights within a network to zero. 
  
![DropConnect](./drop-connect.PNG)
  
### 2. NT-ASGD 
### 3. Extendend regularization techniques 
- Variable length backpropogation sequences 
    - [SortSampler](https://github.com/fastai/fastai/blob/master/fastai/text.py#L118)
    - [SortishSampler](https://github.com/fastai/fastai/blob/master/fastai/text.py#L125)
    - ```bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2```.
    - ```seq_len = max(5, int(np.random.normal(bptt, 5)))```
- Variational dropout 
    - ```Locked Dropout```
- Embedding dropout 
- Weight tying 
- Independent embedding size and hidden size 
- Activation Regulatization (AR) and Temporal Activation Regulaization (TAR)

## Continous Cache Pointers 
[Pointer Cache Models](https://sgugger.github.io/pointer-cache-for-language-model.html#pointer-cache-for-language-model)
- The neural cache model (Grave et al., 2016) can be added on top of a pre-trained language model at negligible cost. The neural cache stores the previous hidden states in memory cells and then uses a simple convex combination of the probability distributions suggested by the cache and the language model for prediction. The cache model has three hyperparameters: the memory size (window) for the cache, the coefficient of the combination (which determines how the two distributions are mixed), and the flatness of the cache distribution. All of these are tuned on the validation set once a trained language model has been obtained and require no training by themselves, making it quite inexpensive to use. 