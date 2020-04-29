# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.


import numpy as np


from contextlib import contextmanager



from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _parasymbolic
else:
    import _parasymbolic

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class ParasymbolicMatrix(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __imul__(self, rhs: "dtype") -> "void":
        r"""
        __imul__(self, rhs)

        Parameters
        ----------
        rhs: dtype

        """
        val = _parasymbolic.ParasymbolicMatrix___imul__(self, rhs)

        return self


        return val


    def __init__(self, size: "size_t", component_count: "size_t"):
        r"""
        __init__(self, size, component_count) -> ParasymbolicMatrix

        Parameters
        ----------
        size: size_t
        component_count: size_t

        """
        _parasymbolic.ParasymbolicMatrix_swiginit(self, _parasymbolic.new_ParasymbolicMatrix(size, component_count))

    def get(self, *args) -> "dtype":
        r"""
        get(self, row, column) -> dtype

        Parameters
        ----------
        row: size_t
        column: size_t

        get(self, comp, row, column) -> dtype

        Parameters
        ----------
        comp: size_t
        row: size_t
        column: size_t

        """
        return _parasymbolic.ParasymbolicMatrix_get(self, *args)

    def total(self) -> "double":
        r"""total(self) -> double"""
        return _parasymbolic.ParasymbolicMatrix_total(self)

    def get_size(self) -> "size_t":
        r"""get_size(self) -> size_t"""
        return _parasymbolic.ParasymbolicMatrix_get_size(self)

    def _prob_any(self, A_v: "dtype const *", A_non_zero_indices: "size_t const *") -> "void":
        r"""
        _prob_any(self, A_v, A_non_zero_indices)

        Parameters
        ----------
        A_v: dtype const *
        A_non_zero_indices: size_t const *

        """
        return _parasymbolic.ParasymbolicMatrix__prob_any(self, A_v, A_non_zero_indices)

    def set_factors(self, A_factors: "dtype const *") -> "void":
        r"""
        set_factors(self, A_factors)

        Parameters
        ----------
        A_factors: dtype const *

        """
        return _parasymbolic.ParasymbolicMatrix_set_factors(self, A_factors)

    def mul_sub_row(self, component: "size_t", row: "size_t", factor: "dtype") -> "void":
        r"""
        mul_sub_row(self, component, row, factor)

        Parameters
        ----------
        component: size_t
        row: size_t
        factor: dtype

        """
        return _parasymbolic.ParasymbolicMatrix_mul_sub_row(self, component, row, factor)

    def mul_sub_col(self, component: "size_t", col: "size_t", factor: "dtype") -> "void":
        r"""
        mul_sub_col(self, component, col, factor)

        Parameters
        ----------
        component: size_t
        col: size_t
        factor: dtype

        """
        return _parasymbolic.ParasymbolicMatrix_mul_sub_col(self, component, col, factor)

    def reset_mul_row(self, component: "size_t", row: "size_t") -> "void":
        r"""
        reset_mul_row(self, component, row)

        Parameters
        ----------
        component: size_t
        row: size_t

        """
        return _parasymbolic.ParasymbolicMatrix_reset_mul_row(self, component, row)

    def reset_mul_col(self, component: "size_t", col: "size_t") -> "void":
        r"""
        reset_mul_col(self, component, col)

        Parameters
        ----------
        component: size_t
        col: size_t

        """
        return _parasymbolic.ParasymbolicMatrix_reset_mul_col(self, component, col)

    def batch_set(self, component_num: "size_t", row: "size_t", A_columns: "size_t const *", A_values: "dtype const *") -> "void":
        r"""
        batch_set(self, component_num, row, A_columns, A_values)

        Parameters
        ----------
        component_num: size_t
        row: size_t
        A_columns: size_t const *
        A_values: dtype const *

        """
        return _parasymbolic.ParasymbolicMatrix_batch_set(self, component_num, row, A_columns, A_values)

    def set_calc_lock(self, value: "bool") -> "void":
        r"""
        set_calc_lock(self, value)

        Parameters
        ----------
        value: bool

        """
        return _parasymbolic.ParasymbolicMatrix_set_calc_lock(self, value)
    __swig_destroy__ = _parasymbolic.delete_ParasymbolicMatrix

    def non_zero_columns(self) -> "std::vector< std::vector< std::vector< size_t > > >":
        r"""non_zero_columns(self) -> std::vector< std::vector< std::vector< size_t > > >"""
        return _parasymbolic.ParasymbolicMatrix_non_zero_columns(self)

    def non_zero_column(self, row_num: "size_t") -> "std::vector< std::vector< size_t > >":
        r"""
        non_zero_column(self, row_num) -> std::vector< std::vector< size_t > >

        Parameters
        ----------
        row_num: size_t

        """
        return _parasymbolic.ParasymbolicMatrix_non_zero_column(self, row_num)

    if '''prob_any''' not in locals():

    	def prob_any(self, v): pass


    if '''__setitem__''' not in locals():

    	def __setitem__(self, key, v): pass


    if '''lock_rebuild''' not in locals():
    	@contextmanager
    	def lock_rebuild(self): pass


# Register ParasymbolicMatrix in _parasymbolic:
_parasymbolic.ParasymbolicMatrix_swigregister(ParasymbolicMatrix)


__temp_store = getattr(ParasymbolicMatrix,"""prob_any""",None)
def __temp_def(self, v):
	nz = np.flatnonzero(v).astype(np.uint64, copy=False)
	return self._prob_any(v, nz)
if isinstance(__temp_store, (classmethod, staticmethod, property)):
	__temp_def = type(__temp_store)(__temp_def)
__temp_def.prev = __temp_store
ParasymbolicMatrix.prob_any = __temp_def
del __temp_store, __temp_def


__temp_store = getattr(ParasymbolicMatrix,"""__setitem__""",None)
def __temp_def(self, key, v):
	comp, row, indices = key
	indices = np.asanyarray(indices, dtype=np.uint64)
	v = np.asanyarray(v, dtype=np.float32)
	self.batch_set(comp, row, indices, v)
if isinstance(__temp_store, (classmethod, staticmethod, property)):
	__temp_def = type(__temp_store)(__temp_def)
__temp_def.prev = __temp_store
ParasymbolicMatrix.__setitem__ = __temp_def
del __temp_store, __temp_def


__temp_store = getattr(ParasymbolicMatrix,"""lock_rebuild""",None)
def __temp_def(self):
	self.set_calc_lock(True)
	yield self
	self.set_calc_lock(False)
__temp_def = contextmanager(__temp_def)
__temp_def.prev = __temp_store
ParasymbolicMatrix.lock_rebuild = __temp_def
del __temp_store, __temp_def



