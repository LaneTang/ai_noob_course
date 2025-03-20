

# 在Mac上部署VScode，创建并启动虚拟环境

**注意**：因vscode下载安装教程在互联****网上已经比较成熟且与Windows系统的安装过程并无大的差异，故本篇就不再介绍。（若有疑问可以参考本次课程给出的Windows的下载安装vscode的教程）


## 第一步：安装Anaconda

* 官方网址：
https://www.anaconda.com/download/success

* 使用清华镜像源下载（下载速度快，推荐）
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?

在网站上输入自己的邮箱地址以注册一个账号，随后网站会将下载网址发送到您的邮箱中。

* 选择下载链接
![alt text](image-5.png)

![alt text](image-8.png)

* 下载好后在访达中打开下载的文件
![alt text](image-1.png)

* 接下来跟着图片所示操作即可
![alt text](image-2.png)

![alt text](image-4.png)

* 若您想自定义安装位置，可以点击"Change Install Location"

![alt text](image-6.png)
>注意安装目录中不要包含中文以及其他特殊字符，即只有英文字母和数字为最佳

* 安装完成后选择“Continue"然后“Close"便可完成安装。您可以自行选择是否删除安装程序
![alt text](image-7.png)


## 第二步：初始化Anaconda

1. **使用终端（Terminal）**
macOS 自带的终端就是你运行 Conda 命令的地方：
   1.  打开 终端（Terminal）
   2.  执行以下命令，确保 Conda 已正确安装,若正确安装，会返回类似 conda 23.x.x 的版本号

```bashq2
conda --version
```
2. **激活 Conda**
安装 Anaconda 后，默认不会自动激活 base 环境，你需要手动激活：

```bash
conda activate base
```

3. **让 Conda 自动加载（可选）**
如果您希望 Conda 在每次打开终端时自动可用，可以执行：

```bash
conda init zsh  # 若使用 zsh（macOS 默认 shell）
conda init bash  # 若使用 bash
```

4. **重新打开终端，就可以直接使用 conda 相关命令了。**


## 第三步：创建环境并安装依赖

1. **创建一个名为 myenv 的环境，并安装 Python 3.12。**
```bash
conda create --name myenv python=3.12
```

2. **激活环境**
 
```bash
conda activate myenv
```

3. 安装其他依赖：
在激活环境后，运行以下命令安装剩余的依赖：
```bash
pip insatll requests
```

```bash
pip insatll openai
```

```bash
pip insatll PyMuPDF
```

```bash
pip insatll jupyter 
```

## 第四步：配置Vscode

按照图中步骤操作即可：

![alt text](image-15.png)

![alt text](image-10.png)

![alt text](image-11.png)

按下 Command + Shift + P 打开命令面板。
输入并选择 “Python: Select Interpreter”。
从列表中选择你的 Conda 环境（例如 /Users/mystline/anaconda3/envs/lanenv/bin/python）。

![alt text](image-12.png)

* 在此界面选择您刚创建的Conda环境
![alt text](image-13.png)

至此，配置就已完成。

