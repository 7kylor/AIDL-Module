### Artificial Intelligence and Deep Learning Module

- Module Code: COMP 20037
- Academic Year: Spring 2024


### Preparation For Python Virtual Environment and Packages Installation In VS Code 

You can choose to install the packages in your global Python environment, or within a virtual environment. It is recommended to use a virtual environment to avoid conflicts with other projects.

# 1. Install Python3

- If you don't have Python3 installed, please install it from the official website:
  [Python3](https://www.python.org/downloads/)
- You can also install Python3 using Homebrew:
 
```sh
 brew install python3
```
# 2. Install Jupyter Notebook

- You can install Jupyter Notebook using pip:

```sh
 pip3 install jupyterlab
```

# 3. Install the Required Packages

- You can install the required packages using pip:

```sh
 pip3 install numpy pandas
```

# 4. Create a Virtual Environment and Install the Packages

- You can create a virtual environment and install the packages using the following
  commands:

```sh
 python3 -m venv {path_to_venv} # Create a virtual environment
 source {path_to_venv}/bin/activate # Activate the virtual environment
 pip3 install {package_name} # Install the package
```

- Please replace `{package_name}` with the name of the package you're trying to
install, and `{path_to_venv}` with the path where you want to create your
virtual environment.

# 5. Install the Required Packages Using a Requirements File

- You can also install the required packages using a requirements file. You can
  create a requirements file by running the following command:

```sh
 pip3 freeze > requirements.txt
```

- You can then install the packages using the following command:

```sh
 pip3 install -r requirements.txt
```

### 6. Install Conda (Optional) >> Recommended -- My Workflow

- You can install Conda by following the instructions on the official website:
  [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  [Conda](https://docs.anaconda.com/free/anaconda/install/mac-os/)

- You can also install Conda using Homebrew:

```sh
  brew install conda
```

- You can also install Conda using pip:

```sh
  pip3 install conda
```

# 7. Create a Conda Environment and Install the Packages

- You can create a Conda environment and install the packages using the following
  commands:

```sh
 conda create --name {env_name} python={version} # Create a Conda environment
 conda activate {env_name} # Activate the Conda environment
 conda install {package_name} # Install the package
```

- Please replace `{env_name}` with the name of the Conda environment you're trying
  to create, `{version}` with the version of Python you want to use, and
  `{package_name}` with the name of the package you're trying to install.

  - I've created a conda environment file called `pyEnv` which contains the required packages for this project.
  - Installed the required packages using the following command:

 - Navigate to the directory where the your project is created and run the following commands:
```sh
  conda env create -n pyEnv  python
  conda activate pyEnv
  conda install numpy pandas jupyter
  ```
  - If you faced any issues with the above command, you can install the packages using the following command:
```sh
  conda install -c conda-forge numpy pandas jupyter
```

# 8. Install the Required Packages Using a Conda Environment File (Optional)

- You can also install the required packages using a Conda environment file. You can
  create a Conda environment file by running the following command:

```sh
 conda env export > environment.yml
```

- You can then install the packages using the following command:

```sh
 conda env create -f environment.yml
```


## 9. Create  a new Jupyter Notebook in VS Code 

- You can create it by pressing `Ctrl + Shift + P` and typing `Jupyter: Create New Blank Notebook` and press `Enter`.

The above steps will help you set up a Python virtual environment and install the required packages in VS Code. You can then start writing your code in a Jupyter Notebook and run the code to see the output.




## .gitignore
/pyEnv/bin*
/pyEnv/etc*
/pyEnv/include*
/pyEnv/lib*
/pyEnv/share*


Conda Documentation — conda 24.1.2 documentation. (n.d.). Retrieved from https://docs.conda.io/en/latest/