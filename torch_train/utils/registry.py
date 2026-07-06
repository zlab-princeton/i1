from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import functools
from collections import abc


def maybe_repeat(arg, n_reps):
    if not isinstance(arg, abc.Sequence):
        arg = (arg,) * n_reps
    return arg


class InKeyOutKey(object):

    def __init__(self, indefault="image", outdefault="image", with_data=False):
        self.indefault = indefault
        self.outdefault = outdefault
        self.with_data = with_data

    def __call__(self, orig_get_pp_fn):

        def get_ikok_pp_fn(*args, key=None,
                            inkey=self.indefault, outkey=self.outdefault, **kw):

            orig_pp_fn = orig_get_pp_fn(*args, **kw)
            def _ikok_pp_fn(data):
                if self.with_data:
                    data[key or outkey] = orig_pp_fn(data[key or inkey], data=data)
                else:
                    data[key or outkey] = orig_pp_fn(data[key or inkey])
                return data

            return _ikok_pp_fn

        return get_ikok_pp_fn


def parse_name(string_to_parse):
    expr = ast.parse(string_to_parse, mode="eval").body
    if not isinstance(expr, (ast.Attribute, ast.Call, ast.Name)):
        raise ValueError(
            "The given string should be a name or a call, but a {} was parsed from "
            "the string {!r}".format(type(expr), string_to_parse))

    if isinstance(expr, ast.Name):
        return string_to_parse, (), {}
    elif isinstance(expr, ast.Attribute):
        return string_to_parse, (), {}

    def _get_func_name(expr):
        if isinstance(expr, ast.Attribute):
            return _get_func_name(expr.value) + "." + expr.attr
        elif isinstance(expr, ast.Name):
            return expr.id
        else:
            raise ValueError(
                "Type {!r} is not supported in a function name, the string to parse "
                "was {!r}".format(type(expr), string_to_parse))

    def _get_func_args_and_kwargs(call):
        args = tuple([ast.literal_eval(arg) for arg in call.args])
        kwargs = {
            kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in call.keywords
        }
        return args, kwargs

    func_name = _get_func_name(expr.func)
    func_args, func_kwargs = _get_func_args_and_kwargs(expr)

    return func_name, func_args, func_kwargs


class Registry(object):

    _GLOBAL_REGISTRY = {}

    @staticmethod
    def global_registry():
        return Registry._GLOBAL_REGISTRY

    @staticmethod
    def register(name, replace=False):

        def _register(item):
            if name in Registry.global_registry() and not replace:
                raise KeyError("The name {!r} was already registered.".format(name))

            Registry.global_registry()[name] = item
            return item

        return _register

    @staticmethod
    def lookup(lookup_string, kwargs_extra=None):

        try:
            name, args, kwargs = parse_name(lookup_string)
        except ValueError as e:
            raise ValueError(f"Error parsing pp:\n{lookup_string}") from e
        if kwargs_extra:
            kwargs.update(kwargs_extra)
        item = Registry.global_registry()[name]
        return functools.partial(item, *args, **kwargs)
