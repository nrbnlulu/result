from result import Err, Ok, Result


def get_result(*, ok: bool) -> Result[str, str]:
	"""Get a test result based on the ok parameter."""
	if ok:
		return Ok("success")
	return Err("error")


def is_int(a: int) -> None:
	"""Assert that a is an integer."""
	assert isinstance(a, int)


def is_float(a: float) -> None:
	"""Assert that a is a float."""
	assert isinstance(a, float)


def is_string(a: str) -> None:
	"""Assert that a is a string."""
	assert isinstance(a, str)


def is_bool(*, a: bool) -> None:
	"""Assert that a is a boolean."""
	assert isinstance(a, bool)


def is_list(a: list) -> None:
	"""Assert that a is a list."""
	assert isinstance(a, list)
