# Calculator

AstrBot 插件 - 计算器

## 介绍

一个数学计算器，支持标量、向量、矩阵的数学表达式解析与计算。使用[Expression-Parser](https://github.com/zyx006/Expression-Parser)作为表达式解析器。

由于插件为Python环境，因而内置裁剪后的win / linux zuluJDK作为[Expression-Parser](https://github.com/zyx006/Expression-Parser)的运行环境，通过jpype启动JVM。

输入/calc help查看使用帮助，/calc help full查看函数与语法。

/calc help
```
===== 计算器帮助 =====
用法：
  /calc <表达式>    计算表达式
  /calc vars        查看变量
  /calc clear       清空变量（不清空 ans）
  /calc help full   查看完整函数与语法列表
内置变量：
  ans   上一次计算结果
```

/calc help full（默认使用群合并转发消息类型，避免刷屏。若非OneBot v11平台则回退普通消息类型）
```
===== 计算器函数与语法 =====
【基本运算】
  +, -, *, /     加减乘除
  %              取模
  ^              幂运算（右结合，如 2^3^2 = 512）
  !              阶乘（如 5! = 120）
【常量】
  PI             圆周率 π ≈ 3.14159...
  E              自然常数 e ≈ 2.71828...

【函数】
  sin, cos, tan          三角函数
  asin, acos, atan       反三角函数
  sinh, cosh, tanh       双曲函数
  exp, ln, log, log10    指数与对数
  sqrt, cbrt, abs        根号与绝对值
  ceil, floor, round     取整函数
  signum, sign           符号函数 (-1, 0, 1)
  degrees, radians       角度 / 弧度转换
  atan2(y,x), hypot(x,y) 双参数反正切 / 斜边
  pow(base, exponent)    幂

【向量 / 数组统计】
  对于向量 X = [...]; Y = [...];(也可手动按序展开输入)

  max(X), min(X)         多参数极值
  sum(X), count(X)       求和 / 参数计数
  avg(X), median(X)      平均值 / 中位数
  prod(X), product(X)    乘积

  gcd(X), lcm(X)         最大公约数 / 最小公倍数
  range(X), geomean(X)   极差 / 几何平均数
  norm1(X), sumabs(X)    L1 范数（绝对值和）
  norm2(X), rms(X)       L2 范数 / 均方根

  var(X), variance(X)    样本方差
  std(X), stddev(X)      样本标准差
  varp(X), variancep(X)  总体方差
  stdp(X), stddevp(X)    总体标准差

  percentile(p, X)       百分位数 (p ∈ [0,100])
  cov(X, Y)              样本协方差
  covp(X, Y)             总体协方差
  corr(X, Y)             相关系数
  dot(X, Y)              向量点积
  dist(X, Y)             欧几里得距离
  manhattan(X, Y)        曼哈顿距离

【矩阵】
  transpose(M), t(M)     向量/矩阵转置
  det(M)                 行列式
  matmul(A, B)           矩阵乘法
  trace(M)               矩阵的迹(主对角线之和)
  rank(M)                矩阵的秩
  mean(M, axis)          矩阵均值(axis=0 列, axis=1 行)
  inv(M)                 矩阵求逆
  solve(A, b)            解线性方程组 Ax=b

【组合数学】
  C(n,k), comb(n,k)      组合数
  P(n,k), perm(n,k)      排列数

【变量】
  x = 10                 赋值
  x = 10; y = 2x         多语句（分号分隔）
  ans                    上一次计算结果

【向量与矩阵】
  [1, 2, 3]              向量
  [[1,2], [3,4]]         矩阵
  scores = [1,2,3]       向量变量赋值
  avg(scores)            向量作为函数参数

【隐式乘法】
  2PI, 3(4+5), 2sqrt(4)
```

## 支持

- [插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)
