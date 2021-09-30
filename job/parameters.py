"""
Infrastructure for managing classes with parameters

One aim is to allow easy management of inheritance of parameters.

.. autosummary::
   :nosignatures:

   Parameter
   DeprecatedParameter
   HideParameter
   Parameterized
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib
import logging
from collections import OrderedDict
from typing import Any, Dict, Sequence, Union

import numpy as np


class NoValueType:
    """special value to indicate no value for a parameter"""

    def __repr__(self):
        return "NoValue"


NoValue = NoValueType()


def auto_type(value):
    """convert value to float or int if reasonable"""
    try:
        float_val = float(value)
    except TypeError:
        return value

    try:
        int_val = int(value)
    except ValueError:
        return float_val

    if int_val == float_val:
        return int_val
    else:
        return float_val


def import_class(identifier: str):
    """import a class or module given an identifier

    Args:
        identifier (str):
            The identifier can be a module or a class. For instance, calling the
            function with the string `identifier == 'numpy.linalg.norm'` is
            roughly equivalent to running `from numpy.linalg import norm` and
            would return a reference to `norm`.
    """
    module_path, _, class_name = identifier.rpartition(".")
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # this happens when identifier does not contain a dot
        return importlib.import_module(class_name)


class Parameter:
    """class representing a single parameter"""

    def __init__(
        self,
        name: str,
        default_value=None,
        cls=object,
        description: str = "",
        *,
        hidden: bool = False,
        extra: Dict[str, Any] = None,
    ):
        """initialize a parameter

        Args:
            name (str):
                The name of the parameter
            default_value:
                The default value
            cls:
                The type of the parameter, which is used for conversion
            description (str):
                A string describing the impact of this parameter. This
                description appears in the parameter help
            hidden (bool):
                Whether the parameter is hidden in the description summary
            extra (dict):
                Extra arguments that are stored with the parameter
        """
        self.name = name
        self.default_value = default_value
        self.cls = cls
        self.description = description
        self.hidden = hidden
        self.extra = {} if extra is None else extra

        if cls is not object and default_value not in {None, NoValue}:
            # check whether the default value is of the correct type
            try:
                converted_value = cls(default_value)
            except TypeError as err:
                raise TypeError(
                    f"Parameter {self.name} has invalid default value: {default_value}"
                ) from err
            if isinstance(converted_value, np.ndarray):
                valid_default = np.allclose(converted_value, default_value)
            else:
                valid_default = converted_value == default_value
            if not valid_default:
                logging.warning(
                    f"Default value `{name}` is not of type `{cls.__name__}`"
                )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(name="{self.name}", default_value='
            f"{self.default_value}, cls={self.cls.__name__}, description="
            f'"{self.description}", required={self.required}, hidden={self.hidden})'
        )

    __str__ = __repr__

    def __getstate__(self):
        # replace the object class by its class path
        return {
            "name": str(self.name),
            "default_value": self.convert(),
            "cls": object.__module__ + "." + self.cls.__name__,
            "description": self.description,
            "required": self.required,
            "hidden": self.hidden,
            "extra": self.extra,
        }

    def __setstate__(self, state):
        # restore the object from the class path
        state["cls"] = import_class(state["cls"])
        # restore the state
        self.__dict__.update(state)

    def convert(self, value=NoValue, *, strict: bool = True):
        """converts a `value` into the correct type for this parameter. If
        `value` is not given, the default value is converted.

        Note that this does not make a copy of the values, which could lead to
        unexpected effects where the default value is changed by an instance.

        Args:
            value:
                The value to convert
            strict (bool):
                Flag indicating whether conversion to the type indicated by `cls` is
                enforced. If `False`, the original value is returned when conversion
                fails.

        Returns:
            The converted value, which is of type `self.cls`
        """
        if value is NoValue:
            value = self.default_value

        if value in {NoValue, None}:
            pass  # treat these values special
        elif self.cls is object:
            value = auto_type(value)
        else:
            try:
                value = self.cls(value)
            except (TypeError, ValueError) as err:
                if strict:
                    raise ValueError(
                        f"Could not convert {value!r} to {self.cls.__name__} for "
                        f"parameter '{self.name}'"
                    ) from err
                # else: just return the value unchanged

        return value

    def _argparser_add(self, parser):
        """add a command line option for this parameter to a parser"""
        if not self.hidden:

            if self.description:
                description = self.description
            else:
                description = f"Parameter `{self.name}`"

            arg_name = "--" + self.name
            kwargs = {
                "default": self.default_value,
                "help": description,
                "metavar": "VALUE",
            }

            if self.cls is bool and self.default_value == False:
                parser.add_argument(
                    arg_name, default=False, action="store_true", **kwargs
                )

            elif self.cls is object or self.cls is auto_type:
                parser.add_argument(arg_name, **kwargs)

            else:
                parser.add_argument(arg_name, type=self.cls, **kwargs)


class DeprecatedParameter(Parameter):
    """a parameter that can still be used normally but is deprecated"""

    pass


class HideParameter:
    """a helper class that allows hiding parameters of the parent classes"""

    def __init__(self, name: str):
        """
        Args:
            name (str):
                The name of the parameter
        """
        self.name = name

    def _argparser_add(self, parser):
        pass


class hybridmethod:
    """
    descriptor that can be used as a decorator to allow calling a method both
    as a classmethod and an instance method

    Adapted from https://stackoverflow.com/a/28238047
    """

    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = doc or fclass.__doc__
        # support use on abstract base classes
        self.__isabstractmethod__ = bool(getattr(fclass, "__isabstractmethod__", False))

    def classmethod(self, fclass):
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            # either bound to the class, or no instance method available
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)


ParameterListType = Sequence[Union[Parameter, HideParameter]]


class Parameterized:
    """a mixin that manages the parameters of a class"""

    parameters_default: ParameterListType = []

    def __init__(self, parameters: Dict[str, Any] = None, *, strict: bool = True):
        """initialize the parameters of the object

        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults. The allowed
                parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
            strict (bool):
                Flag indicating whether parameters are strictly interpreted. If `True`,
                only parameters listed in `parameters_default` can be set and their type
                will be enforced.
        """
        # set logger if this has not happened, yet
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(self.__class__.__name__)

        # set parameters if they have not been initialized, yet
        if not hasattr(self, "parameters"):
            self.parameters = self._parse_parameters(
                parameters, include_deprecated=True, check_validity=strict
            )

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclasses to reconstruct them later"""
        # normalize the parameters_default attribute
        if hasattr(cls, "parameters_default") and isinstance(
            cls.parameters_default, dict
        ):
            # default parameters are given as a dictionary
            cls.parameters_default = [
                Parameter(*args) for args in cls.parameters_default.items()
            ]

    @classmethod
    def get_parameters(
        cls,
        include_hidden: bool = False,
        include_deprecated: bool = False,
        sort: bool = True,
    ) -> Dict[str, Parameter]:
        """return a dictionary of parameters that the class supports

        Args:
            include_hidden (bool): Include hidden parameters
            include_deprecated (bool): Include deprecated parameters
            sort (bool): Return ordered dictionary with sorted keys

        Returns:
            dict: a dictionary of instance of :class:`Parameter` with their
            names as keys.
        """
        # collect the parameters from the class hierarchy
        parameters: Dict[str, Parameter] = {}
        for cls in reversed(cls.__mro__):
            if hasattr(cls, "parameters_default"):
                for p in cls.parameters_default:
                    if isinstance(p, HideParameter):
                        if include_hidden:
                            parameters[p.name].hidden = True
                        else:
                            del parameters[p.name]

                    else:
                        parameters[p.name] = p

        # filter parameters based on hidden and deprecated flags
        def show(p):
            """helper function to decide whether parameter will be shown"""
            # show based on hidden flag?
            show1 = include_hidden or not p.hidden
            # show based on deprecated flag?
            show2 = include_deprecated or not isinstance(p, DeprecatedParameter)
            return show1 and show2

        # filter parameters based on `show`
        result = {
            name: parameter for name, parameter in parameters.items() if show(parameter)
        }

        if sort:
            result = OrderedDict(sorted(result.items()))
        return result

    @classmethod
    def _parse_parameters(
        cls,
        parameters: Dict[str, Any] = None,
        check_validity: bool = True,
        allow_hidden: bool = True,
        include_deprecated: bool = False,
    ) -> Dict[str, Any]:
        """parse parameters

        Args:
            parameters (dict):
                A dictionary of parameters that will be parsed.
            check_validity (bool):
                Determines whether a `ValueError` is raised if there are keys in
                parameters that are not in the defaults. If `False`, additional
                items are simply stored in `self.parameters`
            allow_hidden (bool):
                Allow setting hidden parameters
            include_deprecated (bool):
                Include deprecated parameters
        """
        if parameters is None:
            parameters = {}
        else:
            parameters = parameters.copy()  # do not modify the original

        # obtain all possible parameters
        param_objs = cls.get_parameters(
            include_hidden=allow_hidden, include_deprecated=include_deprecated
        )

        # initialize parameters with default ones from all parent classes
        result: Dict[str, Any] = {}
        for name, param_obj in param_objs.items():
            if not allow_hidden and param_obj.hidden:
                continue  # skip hidden parameters
            # take value from parameters or set default value
            value = parameters.pop(name, NoValue)
            # convert parameter to correct type
            result[name] = param_obj.convert(value, strict=check_validity)

        # update parameters with the supplied ones
        if check_validity and parameters:
            raise ValueError(
                f"Parameters `{sorted(parameters.keys())}` were provided for an "
                f"instance but are not defined for the class `{cls.__name__}`"
            )
        else:
            result.update(parameters)  # add remaining parameters

        return result

    def get_parameter_default(self, name):
        """return the default value for the parameter with `name`

        Args:
            name (str): The parameter name
        """
        for cls in self.__class__.__mro__:
            if hasattr(cls, "parameters_default"):
                for p in cls.parameters_default:
                    if isinstance(p, Parameter) and p.name == name:
                        return p.default_value

        raise KeyError(f"Parameter `{name}` is not defined")

    @classmethod
    def _show_parameters(
        cls,
        description: bool = None,
        sort: bool = False,
        show_hidden: bool = False,
        show_deprecated: bool = False,
        parameter_values: Dict[str, Any] = None,
    ):
        """private method showing all parameters in human readable format

        Args:
            description (bool):
                Flag determining whether the parameter description is shown. The
                default is to show the description only when we are in a jupyter
                notebook environment.
            sort (bool):
                Flag determining whether the parameters are sorted
            show_hidden (bool):
                Flag determining whether hidden parameters are shown
            show_deprecated (bool):
                Flag determining whether deprecated parameters are shown
            parameter_values (dict):
                A dictionary with values to show. Parameters not in this
                dictionary are shown with their default value.

        All flags default to `False`.
        """
        # set the templates for displaying the data
        template = "{name}: {type} = {value!r}"
        template_object = "{name} = {value!r}"
        if description:
            template += " ({description})"
            template_object += " ({description})"

        # iterate over all parameters
        params = cls.get_parameters(
            include_hidden=show_hidden, include_deprecated=show_deprecated, sort=sort
        )
        for param in params.values():
            # initialize the data to show
            data = {
                "name": param.name,
                "type": param.cls.__name__,
                "description": param.description,
            }

            # determine the value to show
            if parameter_values is None:
                data["value"] = param.default_value
            else:
                data["value"] = parameter_values[param.name]

            # print the data to stdout
            if param.cls is object:
                print(template_object.format(**data))
            else:
                print(template.format(**data))

    @hybridmethod
    def show_parameters(  # @NoSelf
        cls,
        description: bool = None,  # @NoSelf
        sort: bool = False,
        show_hidden: bool = False,
        show_deprecated: bool = False,
    ):
        """show all parameters in human readable format

        Args:
            description (bool):
                Flag determining whether the parameter description is shown. The
                default is to show the description only when we are in a jupyter
                notebook environment.
            sort (bool):
                Flag determining whether the parameters are sorted
            show_hidden (bool):
                Flag determining whether hidden parameters are shown
            show_deprecated (bool):
                Flag determining whether deprecated parameters are shown

        All flags default to `False`.
        """
        cls._show_parameters(description, sort, show_hidden, show_deprecated)

    @show_parameters.instancemethod  # type: ignore
    def show_parameters(
        self,
        description: bool = None,  # @NoSelf
        sort: bool = False,
        show_hidden: bool = False,
        show_deprecated: bool = False,
        default_value: bool = False,
    ):
        """show all parameters in human readable format

        Args:
            description (bool):
                Flag determining whether the parameter description is shown. The
                default is to show the description only when we are in a jupyter
                notebook environment.
            sort (bool):
                Flag determining whether the parameters are sorted
            show_hidden (bool):
                Flag determining whether hidden parameters are shown
            show_deprecated (bool):
                Flag determining whether deprecated parameters are shown
            default_value (bool):
                Flag determining whether the default values or the current
                values are shown

        All flags default to `False`.
        """
        self._show_parameters(
            description,
            sort,
            show_hidden,
            show_deprecated,
            parameter_values=None if default_value else self.parameters,
        )
