# KMatch
KM匹配算法实现，匈牙利匹配方法的改进版（但比匈牙利算法早两年提出），可用于大小不一致的二分图匹配算法。可根据给定邻接矩阵计算最小LOSS/最大REWARD时的匹配对。

该函数输入为：邻接矩阵，邻接矩阵中值的类型（LOSS/REWARD）以及无边的值（LOSS/REWARD建议值为32767/0）
输出：临街矩阵列index所匹配到的行index，若返回的行index为-1，则该列无匹配

# Hungarian
匈牙利匹配算法实现

python 版本编译指令
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) KMatch.cpp -o KMatch$(python3-config --extension-suffix)