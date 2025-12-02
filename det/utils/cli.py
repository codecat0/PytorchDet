#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :cli.py
@Author :CodeCat
@Date   :2025/12/2 14:40
"""
import argparse
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml


class ColorTTY(object):
    """
    用于在终端中打印带颜色的文本的工具类。
    支持常用 ANSI 颜色（红、绿、黄、蓝、洋红、青）和加粗样式。
    """

    def __init__(self):
        super(ColorTTY, self).__init__()
        # 定义支持的颜色列表，对应 ANSI 转义码 31~36
        self.colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']

    def __getattr__(self, attr):
        """
        动态支持颜色方法，如 t.red("text")。
        首次调用时生成对应颜色的格式化函数并缓存。
        """
        if attr in self.colors:
            # ANSI 前景色码：31=red, 32=green, ..., 36=cyan
            color_code = self.colors.index(attr) + 31

            def color_message(message):
                # 使用 ANSI 转义序列包裹文本
                return "\033[{}m{}\033[0m".format(color_code, message)

            # 将生成的函数绑定到实例，避免重复创建
            setattr(self, attr, color_message)
            return color_message
        else:
            # 若访问不存在的属性，抛出 AttributeError
            raise AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, attr))

    def bold(self, message):
        """
        将文本设置为加粗样式。

        Args:
            message (str): 要加粗的文本

        Returns:
            str: 包含 ANSI 加粗转义码的字符串
        """
        return self.with_code('01', message)

    @staticmethod
    def with_code(code, message):
        """
        使用指定的 ANSI 代码格式化文本。

        Args:
            code (str): ANSI 转义码（如 '01' 表示加粗，'31' 表示红色）
            message (str): 要格式化的文本

        Returns:
            str: 格式化后的字符串
        """
        return "\033[{}m{}\033[0m".format(code, message)


class ArgsParser(ArgumentParser):
    """
    自定义参数解析器，支持：
      - 从 YAML 配置文件加载基础配置
      - 通过命令行 `-o key=value` 覆盖/添加配置项（支持嵌套，如 `train.lr=0.01`）
    """

    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter
        )
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options"
        )

    def parse_args(self, argv=None):
        """
        解析命令行参数，并验证必要字段。

        Returns:
            argparse.Namespace: 包含 config（str）和 opt（dict）的对象
        """
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    @staticmethod
    def _parse_opt(opts):
        """
        将命令行中的 'key=value' 字符串解析为嵌套字典。

        示例：
          输入: ["train.lr=0.01", "model.name=resnet"]
          输出: {"train": {"lr": 0.01}, "model": {"name": "resnet"}}

        Args:
            opts (list[str]): 命令行传入的 opt 列表

        Returns:
            dict: 解析后的嵌套配置字典
        """
        config = {}
        if not opts:
            return config

        for s in opts:
            s = s.strip()
            if '=' not in s:
                raise ValueError(f"Invalid configuration item: {s}, expected format is key=value.")
            k, v = s.split('=', 1)

            # 安全解析 YAML 值（如 0.01, true, [1,2,3]）
            try:
                parsed_v = yaml.safe_load(v)
            except Exception as e:
                raise ValueError(f"can't parse value '{v}' in '{s}': {e}")

            if '.' not in k:
                # 非嵌套键，直接赋值
                config[k] = parsed_v
            else:
                # 嵌套键（如 train.lr）
                keys = k.split('.')
                current = config
                for idx, key in enumerate(keys[:-1]):
                    if key not in current:
                        current[key] = {}
                    elif not isinstance(current[key], dict):
                        raise ValueError(f"Key conflict: '{k}' - '{key}' is not a dict.")
                    current = current[key]
                current[keys[-1]] = parsed_v

        return config


def merge_args(config, args, exclude_args=['config', 'opt', 'slim_config']):
    for k, v in vars(args).items():
        if k not in exclude_args:
            config[k] = v
    return config
