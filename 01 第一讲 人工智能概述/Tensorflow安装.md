# Tensorflow安装

## 1 安装python包管理工具pip

```bash
$ sudo apt-get install python-dev python-pip
```

## 2 安装tensorflow 

官网http://tflearn.org/找到安装指令
具体查看：http://tflearn.org/installation/
这里安装tensorflow-1.3.0
```
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl
$ sudo pip install $TF_BINARY_URL
```

如果安装失败，用迅雷下载
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl
然后：
```
$ sudo pip install tensorflow-1.3.0-cp27-none-linux_x86_64.whl
```
过程如下：
```
$ sudo pip install tensorflow-1.3.0-cp27-none-linux_x86_64.whl
The directory '/home/fc/.cache/pip/http' or its parent directory is not owned by the current user and the cache has been disabled. Please check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.
The directory '/home/fc/.cache/pip' or its parent directory is not owned by the current user and caching wheels has been disabled. check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.
Processing ./tensorflow-1.3.0-cp27-none-linux_x86_64.whl
Collecting numpy>=1.11.0 (from tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/85/51/ba4564ded90e093dbb6adfc3e21f99ae953d9ad56477e1b0d4a93bacf7d3/numpy-1.15.0-cp27-cp27mu-manylinux1_x86_64.whl (13.8MB)
    100% |████████████████████████████████| 13.8MB 30kB/s 
Collecting mock>=2.0.0 (from tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/e6/35/f187bdf23be87092bd0f1200d43d23076cee4d0dec109f195173fd3ebc79/mock-2.0.0-py2.py3-none-any.whl (56kB)
    100% |████████████████████████████████| 61kB 51kB/s 
Collecting protobuf>=3.3.0 (from tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/b8/c2/b7f587c0aaf8bf2201405e8162323037fe8d17aa21d3c7dda811b8d01469/protobuf-3.6.1-cp27-cp27mu-manylinux1_x86_64.whl (1.1MB)
    100% |████████████████████████████████| 1.1MB 43kB/s 
Requirement already satisfied: wheel in /usr/lib/python2.7/dist-packages (from tensorflow==1.3.0)
Collecting backports.weakref>=1.0rc1 (from tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/88/ec/f598b633c3d5ffe267aaada57d961c94fdfa183c5c3ebda2b6d151943db6/backports.weakref-1.0.post1-py2.py3-none-any.whl
Requirement already satisfied: six>=1.10.0 in /usr/lib/python2.7/dist-packages (from tensorflow==1.3.0)
Collecting tensorflow-tensorboard<0.2.0,>=0.1.0 (from tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/fb/34/14c23665a725c73932891e09b8f017a53aad545c9d5019d2817102dc5d9b/tensorflow_tensorboard-0.1.8-py2-none-any.whl (1.6MB)
    100% |████████████████████████████████| 1.6MB 104kB/s 
Collecting funcsigs>=1; python_version < "3.3" (from mock>=2.0.0->tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/69/cb/f5be453359271714c01b9bd06126eaf2e368f1fddfff30818754b5ac2328/funcsigs-1.0.2-py2.py3-none-any.whl
Collecting pbr>=0.11 (from mock>=2.0.0->tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/69/1c/98cba002ed975a91a0294863d9c774cc0ebe38e05bbb65e83314550b1677/pbr-4.2.0-py2.py3-none-any.whl (100kB)
    100% |████████████████████████████████| 102kB 76kB/s 
Requirement already satisfied: setuptools in /usr/lib/python2.7/dist-packages (from protobuf>=3.3.0->tensorflow==1.3.0)
Collecting bleach==1.5.0 (from tensorflow-tensorboard<0.2.0,>=0.1.0->tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/33/70/86c5fec937ea4964184d4d6c4f0b9551564f821e1c3575907639036d9b90/bleach-1.5.0-py2.py3-none-any.whl
Collecting html5lib==0.9999999 (from tensorflow-tensorboard<0.2.0,>=0.1.0->tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/ae/ae/bcb60402c60932b32dfaf19bb53870b29eda2cd17551ba5639219fb5ebf9/html5lib-0.9999999.tar.gz (889kB)
    100% |████████████████████████████████| 890kB 198kB/s 
Collecting werkzeug>=0.11.10 (from tensorflow-tensorboard<0.2.0,>=0.1.0->tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/20/c4/12e3e56473e52375aa29c4764e70d1b8f3efa6682bef8d0aae04fe335243/Werkzeug-0.14.1-py2.py3-none-any.whl (322kB)
    100% |████████████████████████████████| 327kB 281kB/s 
Collecting markdown>=2.6.8 (from tensorflow-tensorboard<0.2.0,>=0.1.0->tensorflow==1.3.0)
  Downloading https://files.pythonhosted.org/packages/6d/7d/488b90f470b96531a3f5788cf12a93332f543dbab13c423a5e7ce96a0493/Markdown-2.6.11-py2.py3-none-any.whl (78kB)
    100% |████████████████████████████████| 81kB 308kB/s 
Installing collected packages: numpy, funcsigs, pbr, mock, protobuf, backports.weakref, html5lib, bleach, werkzeug, markdown, tensorflow-tensorboard, tensorflow
  Running setup.py install for html5lib ... done
Successfully installed backports.weakref-1.0.post1 bleach-1.5.0 funcsigs-1.0.2 html5lib-0.9999999 markdown-2.6.11 mock-2.0.0 numpy-1.15.0 pbr-4.2.0 protobuf-3.6.1 tensorflow-1.3.0 tensorflow-tensorboard-0.1.8 werkzeug-0.14.1
```

## 3 测试

```
$ python
Python 2.7.15rc1 (default, Apr 15 2018, 21:51:34)
[GCC 7.3.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print tf.__version__
1.3.0
>>>
```
