#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :profiler.py
@Author :CodeCat
@Date   :2025/11/21 16:49
"""
import torch
import torch.profiler as profiler
import sys

# 用于记录调用profiler函数的次数的全局变量
# 用于指定训练步骤的追踪范围
_profiler_step_id = 0

# 用于避免每次都从字符串解析的全局变量
_profiler_options = None
_prof = None


class ProfilerOptions(object):
    """
    使用字符串初始化ProfilerOptions
    字符串格式应为: "key1=value1;key2=value;key3=value3"
    例如:
      "profile_path=model.profile"
      "batch_range=[50, 60]; profile_path=model.profile"
      "batch_range=[50, 60]; tracer_option=OpDetail; profile_path=model.profile"

    ProfilerOptions支持以下键值对:
      batch_range      - 整数列表，例如 [100, 110]
      state            - 字符串，可选值为 'CPU', 'GPU' 或 'All'
      sorted_key       - 字符串，可选值为 'calls', 'total', 'max', 'min' 或 'ave'
      tracer_option    - 字符串，可选值为 'Default', 'OpDetail', 'AllOpDetail'
      profile_path     - 字符串，保存序列化配置文件数据的路径，可用于生成时间线
      exit_on_finished - 布尔值
    """

    def __init__(self, options_str):
        """
        初始化ProfilerOptions

        Args:
            options_str (str): 配置字符串
        """
        assert isinstance(options_str, str)

        # 默认配置
        self._options = {
            'batch_range': [10, 20],
            'state': 'All',
            'sorted_key': 'total',
            'tracer_option': 'Default',
            'profile_path': '/tmp/profile',
            'exit_on_finished': True,
            'timer_only': True
        }
        self._parse_from_string(options_str)

    def _parse_from_string(self, options_str):
        """
        从字符串解析配置选项

        Args:
            options_str (str): 配置字符串
        """
        for kv in options_str.replace(' ', '').split(';'):
            key, value = kv.split('=')
            if key == 'batch_range':
                # 解析批处理范围
                value_list = value.replace('[', '').replace(']', '').split(',')
                value_list = list(map(int, value_list))
                if len(value_list) >= 2 and value_list[0] >= 0 and value_list[1] > value_list[0]:
                    self._options[key] = value_list
            elif key == 'exit_on_finished':
                # 解析退出选项
                self._options[key] = value.lower() in ("yes", "true", "t", "1")
            elif key in [
                'state', 'sorted_key', 'tracer_option', 'profile_path'
            ]:
                # 解析字符串类型的选项
                self._options[key] = value
            elif key == 'timer_only':
                # 解析仅计时选项
                self._options[key] = value

    def __getitem__(self, name):
        """
        获取配置项的值

        Args:
            name (str): 配置项名称

        Returns:
            配置项的值

        Raises:
            ValueError: 当配置项不存在时
        """
        if self._options.get(name, None) is None:
            raise ValueError(
                "ProfilerOptions does not have an option named %s." % name)
        return self._options[name]


def add_profiler_step(options_str=None):
    '''
    使用PyTorch的profiler启用操作级别的计时
    profiler使用独立变量来计算profiler步骤
    此函数的一次调用被视为一个profiler步骤

    Args:
      options_str - 用于初始化ProfilerOptions的字符串
                    默认为None，profiler被禁用
    '''
    if options_str is None:
        return

    global _prof
    global _profiler_step_id
    global _profiler_options

    if _profiler_options is None:
        _profiler_options = ProfilerOptions(options_str)

    # profiler : https://pytorch.org/docs/stable/profiler.html
    # timer_only = True  只显示模型的吞吐量和时间开销
    # timer_only = False 调用summary可以打印从不同角度呈现性能数据的统计表
    # timer_only = False 输出的时间线信息可以在profiler_log目录中找到
    if _prof is None:
        # 创建profiler实例
        _timer_only = str(_profiler_options['timer_only']) == str(True)

        # PyTorch的profiler配置
        _prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA  # 对应GPU
            ],
            schedule=torch.profiler.schedule(
                wait=0,  # 等待前几个step不进行记录
                warmup=0,  # 预热step数
                active=_profiler_options['batch_range'][1] - _profiler_options['batch_range'][0],  # 活跃step数
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_log'),  # 对应export_chrome_tracing
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        _prof.start()
    else:
        _prof.step()

    # 检查是否到达结束step
    if _profiler_step_id == _profiler_options['batch_range'][1]:
        _prof.stop()
        print("Profiler finished. Summary not directly supported in this implementation.")
        print(f"Trace saved to ./profiler_log")
        _prof = None
        if _profiler_options['exit_on_finished']:
            sys.exit(0)

    _profiler_step_id += 1