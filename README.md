# irl-benchmark



## RL module

For our reinforcement learning module (used to train (near) optimal agents for expert demonstrations and as crucial part of the IRL algorithms), we make use of [SLM lab](https://github.com/kengz/SLM-Lab), a modular deep reinforcement learning framework based on PyTorch. Documentation can be found [here](https://kengz.gitbooks.io/slm-lab/content/).

We currently use a fork of this framework, at some point we might make a pull request and use the original repository. To install please do the following steps:

```shell
git clone https://github.com/JohannesHeidecke/SLM-Lab.git
cd SLM-Lab
```

```shell
brew install node yarn
yarn install
```

```shell
conda env update -f environment.yml
pip install -e .
```

