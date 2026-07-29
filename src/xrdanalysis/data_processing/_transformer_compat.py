"""Private rebinding support for the public ``transformers`` facade."""

from __future__ import annotations

from types import FunctionType
from typing import Mapping


def _make_cell(value):
    """Create a closure cell containing ``value``."""
    return (lambda: value).__closure__[0]


def _copy_function_metadata(rebound, function, public_globals):
    """Copy function metadata while retaining the canonical module path."""
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__annotations__ = function.__annotations__
    rebound.__dict__.update(function.__dict__)
    rebound.__doc__ = function.__doc__
    rebound.__module__ = public_globals["__name__"]
    rebound.__qualname__ = function.__qualname__
    if hasattr(function, "__type_params__"):
        rebound.__type_params__ = function.__type_params__
    return rebound


def _rebind_function(function, public_globals: Mapping[str, object]):
    """Clone a method so its implementation uses public module globals."""
    wrapped = getattr(function, "__wrapped__", None)
    if wrapped is not None and function.__closure__ is not None:
        rebound_wrapped = _rebind_function(wrapped, public_globals)
        closure = tuple(
            _make_cell(rebound_wrapped) if cell.cell_contents is wrapped else cell
            for cell in function.__closure__
        )
        rebound = FunctionType(
            function.__code__,
            function.__globals__,
            function.__name__,
            function.__defaults__,
            closure,
        )
        _copy_function_metadata(rebound, function, public_globals)
        rebound.__wrapped__ = rebound_wrapped
        return rebound

    rebound = FunctionType(
        function.__code__,
        public_globals,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    return _copy_function_metadata(rebound, function, public_globals)


def rebind_class_functions(transformer_type, public_globals: Mapping[str, object]):
    """Rebind moved methods to facade globals without changing their API."""
    for name, member in tuple(vars(transformer_type).items()):
        if isinstance(member, staticmethod):
            setattr(
                transformer_type,
                name,
                staticmethod(_rebind_function(member.__func__, public_globals)),
            )
        elif isinstance(member, classmethod):
            setattr(
                transformer_type,
                name,
                classmethod(_rebind_function(member.__func__, public_globals)),
            )
        elif isinstance(member, property):
            setattr(
                transformer_type,
                name,
                property(
                    (
                        _rebind_function(member.fget, public_globals)
                        if member.fget is not None
                        else None
                    ),
                    (
                        _rebind_function(member.fset, public_globals)
                        if member.fset is not None
                        else None
                    ),
                    (
                        _rebind_function(member.fdel, public_globals)
                        if member.fdel is not None
                        else None
                    ),
                    member.__doc__,
                ),
            )
        elif isinstance(member, FunctionType):
            setattr(transformer_type, name, _rebind_function(member, public_globals))
    return transformer_type
