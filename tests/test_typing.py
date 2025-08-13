import math
from typing import Generic, TypeVar

import pytest

from result import Err, Ok, Result, UnwrapError, is_err, is_ok
from tests.utils import get_result, is_float, is_string


def test_union_types_are_good()-> None:
	def needs_result_str_int(res: Result[int | str, str]):
		...
	
	res = get_result(ok=True)
	needs_result_str_int(res)