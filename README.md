# 环境准备

## 安装python
下载地址: https://www.python.org/downloads/

## 虚拟环境管理(conda)
下载地址: https://www.anaconda.com/download

## 安装PyCham
下载地址: https://www.jetbrains.com/zh-cn/pycharm/download/?section=windows

# 设置虚拟环境
conda env create -f environment.yml
conda activate myenv

# 备份虚拟环境
conda env export > environment.yml