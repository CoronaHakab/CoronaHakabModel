# Why are we having this discussion
In some cases we would like to calculate statistics that rise out of other values in Consts, in this case we simply define a function to calculate these values. This brings three issues that often overlap:
* The function is expensive to compute
* The function's output is required to be unique (this is bad by itself, but that's a matter for another day)
* The we want the output to be identical across calls (i.e. `consts.foo() is consts.foo()`)

Just to give an example, currently, Consts's medical state machine is being rebuilt **whenever someone calls `medical_state_machine` or `average_time_in_each_state`**, because someone deleted the caching wrapper because they broke the code.
# lru_cache vs cached_property
In general, there are two ways to solve this problem in the standard library, both of them wrappers in the `functools` module: 
* `lru_cache`: as the name implies, this simply stores the method with a dictionary that caches results by parameters (the dictionary can be created with a maximum size, in which case entries are removed by Least Recently Used order). The downside (for our purposes) is that, for the `self` parameter to be entered as a key, Consts must be Hashable (we will discuss options to do this later)
* `cached_property` this (relatively new) wrapper adds an attribute to each class instance and stores the output there, once called. The main differences are as follows: This wrapper cannot be used on a function with parameters (other than `self`), it does not require the class be hashable, but it does require the class be unslotted.

In truth, the only reason to use `cached_property` in the wild is because `lru_cache` prevents the instances from being collected by the GC, in our code, this is a non-issue since a Consts instance is expected to live through the entire lifetime of the program.
In my opinion, the costs of using `cached_property` (losing `__slots__` for good) far outweigh the benefits hashability can be enforced in many ways, as we will see later.
# Dataclasses vs NamedTuples
Let's get something out of the way first: `NamedTuples` and `dataclasses` are equally readable.
```python
from typing import NamedTuple
from dataclasses import dataclass

class NT(NamedTuple):
    y: bool
    x: int = 0
    ...

@dataclass
class DC:
    y: bool
    x: int = 0
    ...
```
when defining, the two options are equivalent. The primary differences between NT and DC are as follows:
* NT is slotted, DC is not
* NT cannot be extended by a class with non-empty `__slots__`
* Most importantly: **by default, DC is mutable, and therefore, unhashable**

It's important to note that named tuples are only hashable if all its members are hashable, a dataclass is **unhashable unless all its members are explicitly set to be immutable**. 
# Maintaining hashability
At first, requesting that a tuple be guaranteed immutable may seem like a tall order, but there are many ways this can be done with minimal work.
## Hardcode hash-by-id
This can be done with the following snippet:
```python
from typing import NamedTuple

class Foo(NamedTuple):
    ...

    __hash__ = object.__hash__
    __eq__ = object.__eq__
```

This will override regular tuple hash function with the default hash and comparison. It will work but it seems kind of hack-ish. I prefer this solution nevertheless.
## Require Immutable Elements
There's a good reason a tuple is unhashable with unhashable elements, and ideally we should require that consts not include any mutable elements. Any unhashable built-in type, with the sole exception of dicts. This hole can be remedied either by our own hashable dict (not complicated), specialized type per-instance, or using a third-party library.
# What to do with consts
consts.py has bee a thorn at this project's side since it was created. It was meant to be a temporary solution and instead turned into a Glob class. This problem should be fixed and addressed in the future regardless of how we immediately remedy it.