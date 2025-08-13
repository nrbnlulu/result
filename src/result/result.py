from __future__ import annotations

from collections.abc import Awaitable
from typing import (
    Callable,
    Literal,
    NoReturn,
)

from typing_extensions import TypeIs


type Result[T, E] = Ok[T, E] | Err[T, E]


class Ok[T, E]():
    """
    A value that indicates success and which stores arbitrary data for the return value.
    """

    __match_args__ = ("ok_value",)
    __slots__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value: T = value

    def __repr__(self) -> str:  
        return "Ok({})".format(repr(self._value))

    def __eq__(self, other: object) -> bool:  
        return isinstance(other, Ok) and self._value == other._value

    def __ne__(self, other: object) -> bool:  
        return not (self == other)

    def __hash__(self) -> int:  
        return hash((True, self._value))

    def is_ok(self) -> Literal[True]:
        return True

    def is_err(self) -> Literal[False]:
        return False

    def ok(self) -> T:
        """
        Return the value.
        """
        return self._value

    def err(self) -> None:
        """
        Return `None`.
        """
        return None

    @property
    def ok_value(self) -> T:
        """
        Return the inner value.
        """
        return self._value

    @property
    def err_value(self) -> None:
        """
        Return `None` for Ok values.
        """
        return None

    def expect(self, message: str) -> T:
        """
        Return the value.
        """
        return self._value

    def expect_err(self, message: str) -> NoReturn:
        """
        Raise an UnwrapError since this type is `Ok`
        """
        exc = UnwrapError(self, message)
        raise exc

    def unwrap(self) -> T:
        """
        Return the value.
        """
        return self._value

    def unwrap_err(self) -> NoReturn:
        """
        Raise an UnwrapError since this type is `Ok`
        """
        raise UnwrapError(self, "Called `Result.unwrap_err()` on an `Ok` value")

    def unwrap_or[U](self, default: U) -> T:
        """
        Return the value.
        """
        return self._value

    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        """
        Return the value.
        """
        return self._value

    def unwrap_or_raise[Exc: Exception](self, e: type[Exc]) -> T:
        """
        Return the value.
        """
        return self._value

    def map[U](self, op: Callable[[T], U]) -> Ok[U, E]:
        """
        The contained result is `Ok`, so return `Ok` with original value mapped to
        a new value using the passed function.
        """
        return Ok(op(self._value))

    async def map_async[U](
        self, op: Callable[[T], Awaitable[U]]
    ) -> Ok[U, E]:
        """
        The contained result is `Ok`, so return `Ok` with original value mapped to
        a new value using the passed async function.
        """
        return Ok(await op(self._value))

    def map_or[U](self, default: U, op: Callable[[T], U]) -> U:
        """
        The contained result is `Ok`, so return the original value mapped to a new
        value using the passed function.
        """
        return op(self._value)

    def map_or_else[U](self, default_op: Callable[[], U], op: Callable[[T], U]) -> U:
        """
        The contained result is `Ok`, so return the original value mapped to a new
        value using the passed function.
        """
        return op(self._value)

    def map_err[F](self, op: Callable[[E], F]) -> Ok[T, F]:
        """
        The contained result is `Ok`, so return `Ok` with the original value
        """
        return Ok(self._value)

    def and_then[U](self, op: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """
        The contained result is `Ok`, so return the result of `op` with the
        original value passed in
        """
        return op(self._value)

    async def and_then_async[U](
        self, op: Callable[[T], Awaitable[Result[U, E]]]
    ) -> Result[U, E]:
        """
        The contained result is `Ok`, so return the result of `op` with the
        original value passed in
        """
        return await op(self._value)

    def or_else[F](self, op: Callable[[E], Result[T, F]]) -> Ok[T, F]:
        """
        The contained result is `Ok`, so return `Ok` with the original value
        """
        return Ok(self._value)

    def inspect(self, op: Callable[[T], object]) -> Result[T, E]:
        """
        Calls a function with the contained value if `Ok`. Returns the original result.
        """
        _ = op(self._value)
        return self

    async def inspect_async(self, op: Callable[[T], Awaitable[object]]) -> Result[T, E]:
        """
        Calls an async function with the contained value if `Ok`. Returns the original result.
        """
        _ = await op(self._value)
        return self

    def inspect_err(self, op: Callable[[E], object]) -> Result[T, E]:
        """
        Calls a function with the contained value if `Err`. Returns the original result.
        """
        return self

    async def inspect_err_async(self, op: Callable[[E], Awaitable[object]]) -> Result[T, E]:
        """
        Calls an async function with the contained value if `Err`. Returns the original result.
        """
        return self


class DoException[E](Exception):
    """
    This is a special exception used to emulate do-notation, like in Haskell.
    """

    def __init__(self, err: Err[object, E]) -> None:
        super().__init__()
        self.err: Err[object, E] = err


class Err[T, E]:
    """
    A value that signifies failure and which stores arbitrary data for the error.
    """

    __match_args__ = ("err_value",)
    __slots__ = ("_value",)

    def __init__(self, value: E) -> None:
        self._value: E = value

    def __repr__(self) -> str:  
        return "Err({})".format(repr(self._value))

    def __eq__(self, other: object) -> bool:  
        return isinstance(other, Err) and self._value == other._value

    def __ne__(self, other: object) -> bool:  
        return not (self == other)

    def __hash__(self) -> int:  
        return hash((False, self._value))

    def is_ok(self) -> Literal[False]:
        return False

    def is_err(self) -> Literal[True]:
        return True

    def ok(self) -> None:
        """
        Return `None`.
        """
        return None

    def err(self) -> E:
        """
        Return the error.
        """
        return self._value

    @property
    def ok_value(self) -> None:
        """
        Return `None` for Err values.
        """
        return None

    @property
    def err_value(self) -> E:
        """
        Return the inner value.
        """
        return self._value

    def expect(self, message: str) -> NoReturn:
        """
        Raises an `UnwrapError`.
        """
        exc = UnwrapError(self, message)
        raise exc

    def expect_err(self, message: str) -> E:
        """
        Return the inner value
        """
        return self._value

    def unwrap(self) -> NoReturn:
        """
        Raises an `UnwrapError`.
        """
        exc = UnwrapError(self, "Called `Result.unwrap()` on an `Err` value")  # type: ignore[arg-type]
        raise exc

    def unwrap_err(self) -> E:
        """
        Return the inner value
        """
        return self._value

    def unwrap_or[U](self, default: U) -> U:
        """
        Return `default`.
        """
        return default

    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        """
        The contained result is ``Err``, so return the result of applying
        the provided function to the contained value.
        """
        return op(self._value)

    def unwrap_or_raise[Exc: Exception](self, e: type[Exc]) -> NoReturn:
        """
        The contained result is ``Err``, so raise the exception with the value.
        """
        raise e(self._value)

    def map[R](self, op: Callable[[T], R]) -> Err[R, E]:
        """
        Return `Err` with the same value
        """
        return Err(self._value)

    async def map_async[R](self, op: Callable[[T], Awaitable[R]]) -> Err[R, E]:
        """
        Return `Err` with the same value
        """
        return Err(self._value)

    def map_or[R](self, default: R, op: Callable[[T], R]) -> R:
        """
        Return the default value
        """
        return default

    def map_or_else[R](self, default_op: Callable[[], R], op: Callable[[T], R]) -> R:
        """
        Return the result of the default function
        """
        return default_op()

    def map_err[U](self, op: Callable[[E], U]) -> Result[T, U]:
        """
        The contained result is `Err`, so return `Err` with original error mapped to
        a new value using the passed function.
        """
        return Err(op(self._value))

    def and_then[R](self, op: Callable[[T], Result[R, E]]) -> Result[R, E]:
        """
        The contained result is `Err`, so return `Err` with the original value
        """
        return Err(self._value)

    async def and_then_async[R](self, op: Callable[[T], Awaitable[Result[R, E]]]) -> Result[R, E]:
        """
        The contained result is `Err`, so return `Err` with the original value
        """
        return Err(self._value)

    def or_else[U](self, op: Callable[[E], Result[T, U]]) -> Result[T, U]:
        """
        The contained result is `Err`, so return the result of `op` with the
        original value passed in
        """
        return op(self._value)

    def inspect(self, op: Callable[[T], object]) -> Result[T, E]:
        """
        Calls a function with the contained value if `Ok`. Returns the original result.
        """
        return self

    async def inspect_async(self, op: Callable[[T], Awaitable[object]]) -> Result[T, E]:
        """
        Calls an async function with the contained value if `Ok`. Returns the original result.
        """
        return self

    def inspect_err(self, op: Callable[[E], object]) -> Result[T, E]:
        """
        Calls a function with the contained value if `Err`. Returns the original result.
        """
        _ = op(self._value)
        return self

    async def inspect_err_async(self, op: Callable[[E], Awaitable[object]]) -> Result[T, E]:
        """
        Calls an async function with the contained value if `Err`. Returns the original result.
        """
        _ = await op(self._value)
        return self


def is_ok[T, E](result: Result[T, E]) -> TypeIs[Ok[T, E]]:
    """A type guard to check if a result is an Ok

    Arguments:
        result: A result

    Returns:
        A type guard
    """
    return result.is_ok()


def is_err[T, E](result: Result[T, E]) -> TypeIs[Err[T, E]]:
    """A type guard to check if a result is an Err

    Arguments:
        result: A result

    Returns:
        A type guard
    """
    return result.is_err()


class UnwrapError[T, E](Exception):
    """
    Exception raised from ``.unwrap_<...>`` and ``.expect_<...>`` calls.

    The original ``Result`` can be accessed via the ``.result`` attribute, but
    this is not intended for regular use, as type information is lost:
    ``UnwrapError`` doesn't know about both ``T`` and ``E``, since it's raised
    from ``Ok()`` or ``Err()`` which only knows about either ``T`` or ``E``,
    not both.
    """

    def __init__(self, result: Result[T, E], message: str) -> None:
        self._result: Result[T, E] = result
        super().__init__(message)

    @property
    def result(self) -> Result[T, E]:
        """
        Return the original result.
        """
        return self._result
