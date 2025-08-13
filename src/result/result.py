from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
)

from typing_extensions import TypeIs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Define covariant TypeVars for proper subtyping
T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)
R_co = TypeVar("R_co", covariant=True)

class Ok(Generic[T_co, E_co]):
    __match_args__ = ("ok_value",)
    __slots__ = ("_value",)

    def __init__(self, value: T_co) -> None:
        self._value: T_co = value

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

    def ok(self) -> T_co:
        return self._value

    def err(self) -> None:
        return

    @property
    def ok_value(self) -> T_co:
        return self._value

    @property
    def err_value(self) -> None:
        return None

    def expect(self, message: str) -> T_co:
        return self._value

    def expect_err(self, message: str) -> NoReturn:
        exc = UnwrapError(self, message)
        raise exc

    def unwrap(self) -> T_co:
        return self._value

    def unwrap_err(self) -> NoReturn:
        raise UnwrapError(self, "Called `Result.unwrap_err()` on an `Ok` value")

    def unwrap_or[U](self, default: U) -> T_co:
        return self._value

    def unwrap_or_else(self, op: Callable[[E_co], T_co]) -> T_co:
        return self._value

    def unwrap_or_raise[Exc: Exception](self, e: type[Exc]) -> T_co:
        return self._value

    def map[U](self, op: Callable[[T_co], U]) -> Ok[U, E_co]:
        return Ok(op(self._value))

    async def map_async[U](self, op: Callable[[T_co], Awaitable[U]]) -> Ok[U, E_co]:
        return Ok(await op(self._value))

    def map_or[U](self, default: U, op: Callable[[T_co], U]) -> U:
        return op(self._value)

    def map_or_else[U](self, default_op: Callable[[], U], op: Callable[[T_co], U]) -> U:
        return op(self._value)

    def map_err[F](self, op: Callable[[E_co], F]) -> Ok[T_co, F]:
        return Ok(self._value)

    def and_then[R, U](self, op: Callable[[T_co], Result[R, U]]) -> Result[R, U | E_co]:
        return op(self._value)

    async def and_then_async[R, U](
        self, op: Callable[[T_co], Awaitable[Result[R, U]]]
    ) -> Result[R, E_co | U]:
        return await op(self._value)

    def or_else[F](self, op: Callable[[E_co], Result[T_co, F]]) -> Ok[T_co, F]:
        return Ok(self._value)

    def inspect(self, op: Callable[[T_co], object]) -> Result[T_co, E_co]:
        _ = op(self._value)
        return self

    async def inspect_async(self, op: Callable[[T_co], Awaitable[object]]) -> Result[T_co, E_co]:
        _ = await op(self._value)
        return self

    def inspect_err(self, op: Callable[[E_co], object]) -> Result[T_co, E_co]:
        return self

    async def inspect_err_async(
        self, op: Callable[[E_co], Awaitable[object]]
    ) -> Result[T_co, E_co]:
        return self


class Err(Generic[T_co, E_co]):
    __match_args__ = ("err_value",)
    __slots__ = ("_value",)

    def __init__(self, value: E_co) -> None:
        self._value: E_co = value

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

    def err(self) -> E_co:
        return self._value

    @property
    def ok_value(self) -> None:
        return None

    @property
    def err_value(self) -> E_co:
        return self._value

    def expect(self, message: str) -> NoReturn:
        exc = UnwrapError(self, message)
        raise exc

    def expect_err(self, message: str) -> E_co:
        return self._value

    def unwrap(self) -> NoReturn:
        exc = UnwrapError(self, "Called `Result.unwrap()` on an `Err` value")  # type: ignore[arg-type]
        raise exc

    def unwrap_err(self) -> E_co:
        return self._value

    def unwrap_or[U](self, default: U) -> U:
        return default

    def unwrap_or_else(self, op: Callable[[E_co], T_co]) -> T_co:
        return op(self._value)

    def unwrap_or_raise[Exc: Exception](self, e: type[Exc]) -> NoReturn:
        raise e(self._value)

    def map[R](self, op: Callable[[T_co], R]) -> Err[R, E_co]:
        return Err(self._value)

    async def map_async[R](self, op: Callable[[T_co], Awaitable[R]]) -> Err[R, E_co]:
        return Err(self._value)

    def map_or[R](self, default: R, op: Callable[[T_co], R]) -> R:
        return default

    def map_or_else[R](self, default_op: Callable[[], R], op: Callable[[T_co], R]) -> R:
        return default_op()

    def map_err[U](self, op: Callable[[E_co], U]) -> Result[T_co, U]:
        return Err(op(self._value))

    def and_then[R, U](self, op: Callable[[T_co], Result[R, U]]) -> Result[R, U | E_co]:
        return Err(self._value)

    async def and_then_async[R, U](
        self, op: Callable[[T_co], Awaitable[Result[R, U]]]
    ) ->  Result[R, E_co | U]:
        return Err(self._value)

    def or_else[U](self, op: Callable[[E_co], Result[T_co, U]]) -> Result[T_co, U]:
        return op(self._value)

    def inspect(self, op: Callable[[T_co], object]) -> Result[T_co, E_co]:
        return self

    async def inspect_async(self, op: Callable[[T_co], Awaitable[object]]) -> Result[T_co, E_co]:
        return self

    def inspect_err(self, op: Callable[[E_co], object]) -> Result[T_co, E_co]:
        _ = op(self._value)
        return self

    async def inspect_err_async(
        self, op: Callable[[E_co], Awaitable[object]]
    ) -> Result[T_co, E_co]:
        _ = await op(self._value)
        return self


# Define Result as a union of Ok and Err with covariant type parameters
Result = Ok[T_co, E_co] | Err[T_co, E_co]


def is_ok[T, E](result: Result[T, E]) -> TypeIs[Ok[T, E]]:
    return result.is_ok()


def is_err[T, E](result: Result[T, E]) -> TypeIs[Err[T, E]]:
    return result.is_err()


class UnwrapError(Exception, Generic[T_co, E_co]):
    def __init__(self, result: Result[T_co, E_co], message: str) -> None:
        self._result: Result[T_co, E_co] = result
        super().__init__(message)

    @property
    def result(self) -> Result[T_co, E_co]:
        return self._result
