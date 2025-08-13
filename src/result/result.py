from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    NoReturn,
)

from typing_extensions import TypeIs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

type Result[T, E] = Ok[T, E] | Err[T, E]


class Ok[T, E]:
    __match_args__ = ("ok_value",)
    __slots__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value: T = value

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"

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
        return self._value

    def err(self) -> None:
        return

    @property
    def ok_value(self) -> T:
        return self._value

    @property
    def err_value(self) -> None:
        return None

    def expect(self, message: str) -> T:
        return self._value

    def expect_err(self, message: str) -> NoReturn:
        exc = UnwrapError(self, message)
        raise exc

    def unwrap(self) -> T:
        return self._value

    def unwrap_err(self) -> NoReturn:
        raise UnwrapError(self, "Called `Result.unwrap_err()` on an `Ok` value")

    def unwrap_or[U](self, default: U) -> T:
        return self._value

    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        return self._value

    def unwrap_or_raise[Exc: Exception](self, e: type[Exc]) -> T:
        return self._value

    def map[U](self, op: Callable[[T], U]) -> Ok[U, E]:
        return Ok(op(self._value))

    async def map_async[U](self, op: Callable[[T], Awaitable[U]]) -> Ok[U, E]:
        return Ok(await op(self._value))

    def map_or[U](self, default: U, op: Callable[[T], U]) -> U:
        return op(self._value)

    def map_or_else[U](self, default_op: Callable[[], U], op: Callable[[T], U]) -> U:
        return op(self._value)

    def map_err[F](self, op: Callable[[E], F]) -> Ok[T, F]:
        return Ok(self._value)

    def and_then[U](self, op: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return op(self._value)

    async def and_then_async[U](self, op: Callable[[T], Awaitable[Result[U, E]]]) -> Result[U, E]:
        return await op(self._value)

    def or_else[F](self, op: Callable[[E], Result[T, F]]) -> Ok[T, F]:
        return Ok(self._value)

    def inspect(self, op: Callable[[T], object]) -> Result[T, E]:
        _ = op(self._value)
        return self

    async def inspect_async(self, op: Callable[[T], Awaitable[object]]) -> Result[T, E]:
        _ = await op(self._value)
        return self

    def inspect_err(self, op: Callable[[E], object]) -> Result[T, E]:
        return self

    async def inspect_err_async(self, op: Callable[[E], Awaitable[object]]) -> Result[T, E]:
        return self


class DoException[E](Exception):
    def __init__(self, err: Err[object, E]) -> None:
        super().__init__()
        self.err: Err[object, E] = err


class Err[T, E]:
    __match_args__ = ("err_value",)
    __slots__ = ("_value",)

    def __init__(self, value: E) -> None:
        self._value: E = value

    def __repr__(self) -> str:
        return f"Err({self._value!r})"

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
        return

    def err(self) -> E:
        return self._value

    @property
    def ok_value(self) -> None:
        return None

    @property
    def err_value(self) -> E:
        return self._value

    def expect(self, message: str) -> NoReturn:
        exc = UnwrapError(self, message)
        raise exc

    def expect_err(self, message: str) -> E:
        return self._value

    def unwrap(self) -> NoReturn:
        exc = UnwrapError(self, "Called `Result.unwrap()` on an `Err` value")  # type: ignore[arg-type]
        raise exc

    def unwrap_err(self) -> E:
        return self._value

    def unwrap_or[U](self, default: U) -> U:
        return default

    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        return op(self._value)

    def unwrap_or_raise[Exc: Exception](self, e: type[Exc]) -> NoReturn:
        raise e(self._value)

    def map[R](self, op: Callable[[T], R]) -> Err[R, E]:
        return Err(self._value)

    async def map_async[R](self, op: Callable[[T], Awaitable[R]]) -> Err[R, E]:
        return Err(self._value)

    def map_or[R](self, default: R, op: Callable[[T], R]) -> R:
        return default

    def map_or_else[R](self, default_op: Callable[[], R], op: Callable[[T], R]) -> R:
        return default_op()

    def map_err[U](self, op: Callable[[E], U]) -> Result[T, U]:
        return Err(op(self._value))

    def and_then[R](self, op: Callable[[T], Result[R, E]]) -> Result[R, E]:
        return Err(self._value)

    async def and_then_async[R](self, op: Callable[[T], Awaitable[Result[R, E]]]) -> Result[R, E]:
        return Err(self._value)

    def or_else[U](self, op: Callable[[E], Result[T, U]]) -> Result[T, U]:
        return op(self._value)

    def inspect(self, op: Callable[[T], object]) -> Result[T, E]:
        return self

    async def inspect_async(self, op: Callable[[T], Awaitable[object]]) -> Result[T, E]:
        return self

    def inspect_err(self, op: Callable[[E], object]) -> Result[T, E]:
        _ = op(self._value)
        return self

    async def inspect_err_async(self, op: Callable[[E], Awaitable[object]]) -> Result[T, E]:
        _ = await op(self._value)
        return self


def is_ok[T, E](result: Result[T, E]) -> TypeIs[Ok[T, E]]:
    return result.is_ok()


def is_err[T, E](result: Result[T, E]) -> TypeIs[Err[T, E]]:
    return result.is_err()


class UnwrapError[T, E](Exception):
    def __init__(self, result: Result[T, E], message: str) -> None:
        self._result: Result[T, E] = result
        super().__init__(message)

    @property
    def result(self) -> Result[T, E]:
        return self._result
