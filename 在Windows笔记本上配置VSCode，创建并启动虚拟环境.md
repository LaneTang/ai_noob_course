
# 1 - 安装 VSCode + Anaconda

## 下载vscode

### 1. 打开VSCode官网链接:
[Visual Studio Code - Code Editing. Redefined](https://code.visualstudio.com/)

![[Pasted image 20250309154601.png]]

### 2. 点击Download for Windows

可以看到已经开始下载了

![[Pasted image 20250309154657.png]]

## 安装VSCode

点击打开文件

![[Pasted image 20250309154950.png]]

选择 "我同意此协议", 下一步

![[Pasted image 20250309155054.png]]

将路径替换到想放的地方, 下一步

![[Pasted image 20250309155510.png]]

默认, 下一步

![[Pasted image 20250309155546.png]]

勾上 "创建桌面快捷方式", 下一步

![[Pasted image 20250309155040.png]]

安装

![[Pasted image 20250309155156.png]]

完成并打开vscode！

![[Pasted image 20250309155713.png]]

## 下载Anaconda

Anaconda 是一个流行的Python数据科学平台，Anaconda 可以看做Python的一个集成安装，安装它后就默认安装了python、IPython、集成开发环境Spyder和众多的包和模块，让你在管理环境和包时更加方便。

可以选择从官网或清华源下载 Anaconda 安装包，**清华源**下载速度相对较快。

#### 1、从官网下载

打开官网[Download Now | Anaconda](https://www.anaconda.com/download/success)

![[Pasted image 20250309160133.png]]

往下翻，选择windows安装包，点击下载

![[Pasted image 20250309160157.png]]

#### 2、使用清华镜像源下载（下载速度快，推荐）

[Anaconda清华镜像源下载](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D "Anaconda清华镜像源下载")

![[Pasted image 20250309160347.png]]

这里选择下载最新版本的

![[Pasted image 20250309160439.png]]

## 安装Anaconda

下载完成我们就可以得到一个exe文件，点击运行

![[Pasted image 20250309161524.png]]

一直next，到这里，把anaconda安装目录改成自己的目录，next

默认，直接install

![[Pasted image 20250309161656.png]]

# 2 - 配置Anaconda

打开 **Anaconda prompt**

![[Pasted image 20250309162419.png]]

找到我们给的requirements.txt，复制requirements.txt的路径。

环境的搭建有两种方法, 一种是使用conda源进行安装, 一种是使用pip源进行安装, ==本次课程我们使用pip源安装作为教程==
### 创建python虚拟环境

首先创建虚拟环境，复制粘贴下面的命令，回车。
这表示创建python版本为3.12、名字为env_name的虚拟环境。

```bash
conda create -n env_name python=3.12 # env_name替换成想要的名字
```

然后激活我们创建的conda环境

```bash
conda activate env_name # env_name替换成想要的名字
```

![[Pasted image 20250309164139.png]]

## pip源添加

##  方法一：临时指定镜像源（单次生效）

背景：有些软件包需要使用pip安装而不能使用conda安装，但**Anaconda中的pip**没国内镜像源的配置，会导致下载错误。

没有使用国内镜像源而导致的pip下载错误

在 `pip install` 命令中直接添加 `-i` 参数指定镜像源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

## 方法二：永久修改 pip 源

创建或修改 pip 配置文件，设置默认镜像源。

- **Windows 系统**：
	1. 打开文件资源管理器，输入路径：`%APPDATA%`，进入此目录。
	2. 新建文件夹 `pip`（若不存在），并在其中新建文件 `pip.ini`。
	3. 编辑 `pip.ini`，添加以下内容：

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

复制粘贴下面的命令，回车
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r D:\workSpace\pyProj\requirements.txt # 这里替换成你requirements.txt的实际路径
```

![[Pasted image 20250309171318.png]]

或者也可以选择手动安装

```bash
# 创建py环境
(base) C:\Users\tanga>pip create -n my_env python=3.12 # my_env 记得替换

# 激活启动对应的环境
(base) C:\Users\tanga>pip activate my_env # my_env 记得替换

# 安装对应需要的包
pip install requests

pip install jupyter

pip install openai

pip install PyMuPDF
```

## 配置VSCode

打开VSCode -> 左栏extension

![[Pasted image 20250309165452.png]]

搜索 **python**，安装（install）

![[Pasted image 20250309165528.png]]

再次打开 **Anaconda prompt**

输入 `conda env list`，找到之前安装的环境和它对应的路径地址，看看自己新装的环境的地址在哪

![[Pasted image 20250309165846.png]]

在 VSCode 中按 `Ctrl+Shift+P`，输入 `Python: Select Interpreter` 并选择自己刚刚安装的Conda环境：

![[Pasted image 20250309170042.png]]

OK，环境配置完成！

