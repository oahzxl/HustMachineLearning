把mnist数据集解压放在data文件夹下



命令行运行：

```
正常
python main.py --train --evaluate --k 3 --scale 0.7

显示k近邻的图片
python main.py --train --evaluate --k 3 --plot

显示不同k的acc
python main.py --acc

查看参数说明
python main.py --help
```

