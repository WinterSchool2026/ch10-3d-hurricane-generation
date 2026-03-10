## Installation with pixi

1. Install pixi from terminal: 

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
Close terminal then check if pixi command is recognized.

2. Add pixi to path
```bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

3. create pixi environment. From within environments folder:

```bash
  pixi  install
  ```
4. activate pixi environment. From within environments folder:
```bash
  pixi shell
  ```

5. set evironment in VS code: 
  a. install python + jupyter extensions
  b. ctr+shift+P: Python select interpreter > find python path: /environments/.pixi/envs/default/bin/python


## Installation conda

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the minclouds environment:

```bash
conda env create -f environment.yml

conda activate minclouds
```
