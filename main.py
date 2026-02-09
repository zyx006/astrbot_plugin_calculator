import os
import platform

import jpype
from jpype import JClass

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import logger
from astrbot.core.message.components import Plain, Node


class Calculator(Star):
    evaluator = None
    contexts: dict[str, object] = {} # 存储变量的上下文，允许共用，按 session_id 隔离上下文

    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(base_dir, "Expression-Parser-1.4.0.jar")
        if not os.path.exists(jar_path):
            raise RuntimeError(f"缺失 Jar 文件: {jar_path}")

        if not jpype.isJVMStarted():
            system = platform.system()
            if system == "Windows":  # jlink from zulu jdk25
                JVM_PATH = os.path.join(base_dir, "runtime-win", "bin", "server", "jvm.dll")
            elif system == "Linux":  # jlink from zulu jdk21
                JVM_PATH = os.path.join(base_dir, "runtime-linux", "lib", "server", "libjvm.so")
            else:
                JVM_PATH = jpype.getDefaultJVMPath()

            logger.info(f"Using jar: {jar_path}")
            jpype.startJVM(JVM_PATH, "--enable-native-access=ALL-UNNAMED", classpath=[jar_path])

        try:
            ExpressionEvaluator = JClass("cn.czyx007.expression_parser.api.ExpressionEvaluator")
        except jpype.JException as ex:
            logger.error(f"无法加载 ExpressionEvaluator，请确保 {jar_path} 在 JVM classpath 中")
            raise RuntimeError(f"Calculator插件初始化失败: {ex}")
        self.evaluator = ExpressionEvaluator()

    @filter.command("calc")
    async def calculator(self, event: AstrMessageEvent):
        """这是一个计算器插件，支持标量、向量、矩阵的数学表达式解析与计算。
        输入/calc help查看使用帮助，/calc help full查看函数与语法。"""
        message_str = event.message_str
        cmd, _, expr = message_str.partition(" ")
        if expr == "help":
            yield event.plain_result(self.help_main())
            return
        if expr == "help full":
            if event.get_platform_name() == "aiocqhttp":
                # onebot平台发送群合并消息
                node = Node(
                    uin=event.get_self_id(),
                    name=".",
                    content=[
                        Plain(self.help_full()),
                    ]
                )
                yield event.chain_result([node])
                return
            else:# 其他平台直接发送纯文本
                yield event.plain_result(self.help_full())
                return
        if self.evaluator is None or self.context is None:
            yield event.plain_result("计算器插件未正确初始化，请联系管理员重启bot。")
            return

        # 获取独立上下文
        session_id = event.get_session_id()
        if session_id not in self.contexts:
            HashMap = JClass("java.util.HashMap")
            self.contexts[session_id] = HashMap()
        context = self.contexts[session_id]

        # vars 只输出ans以外的变量
        if expr == "vars":
            if context.isEmpty() or context.size() <= 1:
                yield event.plain_result("当前没有任何变量。")
                return

            lines = []
            entry_set = context.entrySet()
            for entry in entry_set:
                if entry.getKey() != "ans":
                    lines.append(f"{entry.getKey()} = {entry.getValue()}")

            yield event.plain_result("\n".join(lines))
            return
        # clear
        if expr == "clear":
            ans = context.get("ans")
            context.clear()
            if ans is not None:
                context.put("ans", ans)
            yield event.plain_result("变量已清空。")
            return
        # 正常计算
        try:
            result = self.evaluator.eval(expr, context)
            context.put("ans", result)
            yield event.plain_result(str(result))
        except Exception as e:
            err_msg = str(e).split(": ", 1)[1] if ": " in str(e) else str(e)
            yield event.plain_result(f"计算错误：{err_msg}")

    def help_main(self) -> str:
        return """\
===== 计算器帮助 =====
用法：
  /calc <表达式>    计算表达式
  /calc vars        查看变量
  /calc clear       清空变量（不清空 ans）
  /calc help full   查看完整函数与语法列表
内置变量：
  ans   上一次计算结果
"""

    def help_full(self) -> str:
        return """\
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
"""

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
        #不关闭JVM，因为整个进程周期只能初始化一次JVM，关闭后无法重启JVM