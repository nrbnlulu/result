from collections.abc import Sequence

from result import Err, Ok, Result
from tests.utils import get_result, is_int


def test_union_types_are_good() -> None:
	def needs_result_str_int(res: Result[int | str, str]) -> None: ...

	res: Result[str, str] = get_result(ok=True)
	needs_result_str_int(res)


def test_ok_covariance() -> None:
	"""Test that Ok types are covariant in their success type."""

	def accepts_ok_union(ok_val: Ok[int | str, str]) -> None: ...

	# str is a subtype of int | str, so this should work
	ok_str: Ok[str, str] = Ok("hello")
	accepts_ok_union(ok_str)

	# int is a subtype of int | str, so this should work too
	ok_int: Ok[int, str] = Ok(42)
	accepts_ok_union(ok_int)


def test_err_covariance() -> None:
	"""Test that Err types are covariant in their error type."""

	def accepts_err_union(err_val: Err[str, int | str]) -> None: ...

	# str is a subtype of int | str, so this should work
	err_str: Err[str, str] = Err("error")
	accepts_err_union(err_str)

	# int is a subtype of int | str, so this should work too
	err_int: Err[str, int] = Err(42)
	accepts_err_union(err_int)


def test_result_covariance_comprehensive() -> None:
	"""Test comprehensive covariance for Result types."""

	def accepts_broad_result(res: Result[int | str | float, int | str]) -> None: ...

	# All of these should be valid due to covariance
	result_str_str: Result[str, str] = Ok("hello")
	result_int_str: Result[int, str] = Ok(42)
	result_float_str: Result[float, str] = Ok(3.14)
	result_str_int: Result[str, int] = Err(404)
	result_int_int: Result[int, int] = Err(500)

	accepts_broad_result(result_str_str)
	accepts_broad_result(result_int_str)
	accepts_broad_result(result_float_str)
	accepts_broad_result(result_str_int)
	accepts_broad_result(result_int_int)


def test_nested_generic_covariance() -> None:
	"""Test covariance works with nested generic types."""

	def accepts_sequence_result(res: Result[Sequence[int | str], str]) -> None: ...

	result_seq_str: Result[Sequence[str], str] = Ok(["hello", "world"])
	accepts_sequence_result(result_seq_str)

	result_seq_int: Result[Sequence[int], str] = Ok([1, 2, 3])
	accepts_sequence_result(result_seq_int)


def is_int_or_str(a: int | str) -> None: ...


def test_and_then_can_unify_error_types_as_well() -> None:
	def and_then(data: str) -> Result[int, int]:
		return Ok(len(data))

	res = get_result(ok=True).and_then(and_then)
	is_int(res.unwrap())
	is_int_or_str(res.unwrap())


async def test_and_then_async_can_unify_error_types_as_well() -> None:
	async def and_then_async(data: str) -> Result[int, int]:
		return Ok(len(data))

	_ = await get_result(ok=True).and_then_async(and_then_async)
