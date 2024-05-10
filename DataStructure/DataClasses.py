import inspect
import logging
import types
import typing
from dataclasses import KW_ONLY, asdict, dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)


def _find_type_origin(type_hint):
    if isinstance(type_hint, typing._SpecialForm):
        # case of typing.Any, typing.ClassVar, typing.Final, typing.Literal,
        # typing.NoReturn, typing.Optional, or typing.Union without parameters
        return

    actual_type = typing.get_origin(type_hint) or type_hint

    if actual_type is types.UnionType or isinstance(actual_type, typing._SpecialForm):
        # case of typing.Union[…] or typing.ClassVar[…] or …
        for origins in map(_find_type_origin, typing.get_args(type_hint)):
            yield from origins

    else:
        yield actual_type


def _check_types(parameters, hints,
                 convert_int, convert_float, convert_str, convert_types):
    new_kwargs = {}
    for name, value in parameters.items():
        type_hint = hints.get(name, typing.Any)
        actual_types = tuple(_find_type_origin(type_hint))
        logger.debug(f'{name}: {actual_types}')

        if actual_types and not isinstance(value, actual_types):
            converted = False
            for item in convert_types:
                try:
                    if item[0] in actual_types:
                        value = item[1](value)
                        new_kwargs[name] = value
                        converted = True
                        break
                except:
                    pass

            if convert_int:
                try:
                    if int in actual_types:
                        value = int(value)
                        new_kwargs[name] = value
                        converted = True
                        continue
                except:
                    pass

            if convert_float:
                try:
                    if float in actual_types:
                        value = float(value)
                        new_kwargs[name] = value
                        converted = True
                        continue
                except:
                    pass

            if convert_str:
                try:
                    if str in actual_types:
                        value = value.decode()
                        new_kwargs[name] = value
                        converted = True
                        continue
                except (UnicodeDecodeError, AttributeError):
                    pass

            if not converted:
                raise TypeError(
                    f"Expected type '{type_hint}' for argument '{name}'"
                    f" but received type '{type(value)}' instead"
                )

    return new_kwargs


def convert_and_enforce_types(convert_int=True, convert_float=True, convert_str=True,
                              convert_types: list[tuple] = []):
    """_summary_

    Args:
        convert_int (bool, optional): Try to use int(v). Defaults to True.
        convert_float (bool, optional): Try to use float(v). Defaults to True.
        convert_str (bool, optional): Try to use v.decode(). Defaults to True.
        convert_types (list[tuple], optional): Try to convert custom types. Defaults to [].
        Access the format like: [
            (type1, convert_func1),
            (type2, convert_func2),
            ...
        ]
    """
    def class_or_func(callable):
        def decorate(func):
            hints = typing.get_type_hints(func)
            signature = inspect.signature(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                parameters = dict(zip(signature.parameters, args))
                parameters.update(kwargs)
                converted_kwargs = _check_types(parameters, hints,
                                                convert_int, convert_float, convert_str,
                                                convert_types)
                parameters.update(converted_kwargs)

                return func(**parameters)
            return wrapper

        if inspect.isclass(callable):
            callable.__init__ = decorate(callable.__init__)
            return callable

        return decorate(callable)
    return class_or_func


@dataclass
class DataClass:
    _: KW_ONLY
    _extra_fields: list = field(init=False, default_factory=list)

    @classmethod
    def model_validate(cls, kwargs: dict, extra='forbid'):
        """Create from dictionary.

        Args:
            kwargs (dict): _description_
            extra (str, optional): ['forbid', 'ignore', 'allow']. Defaults to 'forbid'.

        Returns:
            dataclass: _description_
        """
        if not extra in ['forbid', 'ignore', 'allow']:
            raise ValueError(
                f"The value of 'extra' must in ['forbid', 'ignore', 'allow']"
                f" but received '{extra}' instead"
            )

        if extra == 'forbid':
            return cls(**kwargs)

        # fetch the constructor's signature
        cls_fields = {field for field in inspect.signature(cls).parameters}

        # split the kwargs into native ones and new ones
        native_args, extra_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                native_args[name] = val
            else:
                extra_args[name] = val

        # use the native ones to create the class ...
        ret = cls(**native_args)

        if extra == 'allow':
            # ... and add the new ones by hand
            extra_fields = []
            for extra_name, extra_val in extra_args.items():
                setattr(ret, extra_name, extra_val)
                extra_fields.append(extra_name)
            ret._extra_fields = extra_fields
        return ret

    def model_dump(self, extra='ignore'):
        """Make dictionary.

        Args:
            extra (str, optional): ['ignore', 'append']. Defaults to 'ignore'.

        Returns:
            _type_: _description_
        """
        if not extra in ['ignore', 'append']:
            raise ValueError(
                f"The value of 'extra' must in ['ignore', 'append']"
                f" but received '{extra}' instead"
            )
        result = asdict(self)
        extra_fields = result.pop('_extra_fields')

        if extra == 'append':
            for name in extra_fields:
                value = getattr(self, name, None)
                if value is None:
                    continue
                result[name] = value
        return result
