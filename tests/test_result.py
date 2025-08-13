import math
from typing import Generic, TypeVar

import pytest

from result import Err, Ok, Result, UnwrapError, is_err, is_ok
from tests.utils import get_result, is_float, is_string


# Basic construction and type checking tests
def test_ok_construction() -> None:
    """Test Ok construction and basic properties."""
    result = Ok("success")
    assert result.is_ok()
    assert not result.is_err()
    assert repr(result) == "Ok('success')"


def test_err_construction() -> None:
    """Test Err construction and basic properties."""
    result = Err("error message")
    assert not result.is_ok()
    assert result.is_err()
    assert repr(result) == "Err('error message')"


def test_pattern_matching_ok() -> None:
    """Test pattern matching with Ok values."""
    result = get_result(ok=True)
    match result:
        case Ok(data):
            is_string(data)
            assert data == "success"
        case Err(_):
            pytest.fail("Should not match Err")


def test_pattern_matching_err() -> None:
    """Test pattern matching with Err values."""
    result = get_result(ok=False)
    match result:
        case Ok(_):
            pytest.fail("Should not match Ok")
        case Err(err):
            is_string(err)
            assert err == "error"


# Unwrap and expect tests
def test_ok_unwrap() -> None:
    """Test Ok unwrap method."""
    result = Ok(42)
    assert result.unwrap() == 42


def test_err_unwrap() -> None:
    """Test Err unwrap method raises UnwrapError."""
    result = Err("error message")
    with pytest.raises(UnwrapError, match="Called `Result.unwrap\\(\\)` on an `Err` value"):
        result.unwrap()


def test_ok_unwrap_err() -> None:
    """Test Ok unwrap_err method raises UnwrapError."""
    result = Ok(42)
    with pytest.raises(UnwrapError, match="Called `Result.unwrap_err\\(\\)` on an `Ok` value"):
        result.unwrap_err()


def test_err_unwrap_err() -> None:
    """Test Err unwrap_err method."""
    result = Err("error message")
    assert result.unwrap_err() == "error message"


def test_ok_expect() -> None:
    """Test Ok expect method."""
    result = Ok(42)
    assert result.expect("Should be Ok") == 42


def test_err_expect() -> None:
    """Test Err expect method raises UnwrapError."""
    result = Err("error message")
    with pytest.raises(UnwrapError, match="Custom message"):
        result.expect("Custom message")


def test_ok_expect_err() -> None:
    """Test Ok expect_err method raises UnwrapError."""
    result = Ok(42)
    with pytest.raises(UnwrapError, match="Should be Err"):
        result.expect_err("Should be Err")


def test_err_expect_err() -> None:
    """Test Err expect_err method."""
    result = Err("error message")
    assert result.expect_err("Should be Err") == "error message"


# unwrap_or and unwrap_or_else tests
def test_ok_unwrap_or() -> None:
    """Test Ok unwrap_or method."""
    result = Ok(42)
    assert result.unwrap_or(0) == 42


def test_err_unwrap_or() -> None:
    """Test Err unwrap_or method."""
    result = Err("error message")
    assert result.unwrap_or(0) == 0


def test_ok_unwrap_or_else() -> None:
    """Test Ok unwrap_or_else method."""
    result = Ok(42)
    assert result.unwrap_or_else(lambda e: len(e)) == 42


def test_err_unwrap_or_else() -> None:
    """Test Err unwrap_or_else method."""
    result = Err("error")
    assert result.unwrap_or_else(lambda e: len(e)) == 5


def test_ok_unwrap_or_raise() -> None:
    """Test Ok unwrap_or_raise method."""
    result = Ok(42)
    assert result.unwrap_or_raise(ValueError) == 42


def test_err_unwrap_or_raise() -> None:
    """Test Err unwrap_or_raise method raises the specified exception."""
    result = Err("error message")
    with pytest.raises(ValueError, match="error message"):
        result.unwrap_or_raise(ValueError)


# Property access tests
def test_ok_properties() -> None:
    """Test Ok property access methods."""
    result = Ok("value")
    assert result.ok() == "value"
    assert result.err() is None
    assert result.ok_value == "value"
    assert result.err_value is None


def test_err_properties() -> None:
    """Test Err property access methods."""
    result = Err("error")
    assert result.ok() is None
    assert result.err() == "error"
    assert result.ok_value is None
    assert result.err_value == "error"


# Map tests with comprehensive type conversion verification
def test_ok_map() -> None:
    """Test Ok map method."""
    result = Ok(5)
    mapped = result.map(lambda x: x * 2)
    assert mapped == Ok(10)


def test_err_map() -> None:
    """Test Err map method preserves error."""
    result = Err("error")
    mapped = result.map(lambda x: x * 2)
    assert mapped == Err("error")


def test_ok_map_type_conversion() -> None:
    """Test that map properly converts types and pyright infers them correctly."""
    # int -> str
    int_result: Result[int, str] = Ok(42)
    str_mapped: Result[str, str] = int_result.map(lambda x: str(x))
    match str_mapped:
        case Ok(value):
            is_string(value)
            assert value == "42"
        case Err(_):
            pytest.fail("Should be Ok")

    # str -> bool
    str_result: Result[str, int] = Ok("hello")
    bool_mapped: Result[bool, int] = str_result.map(lambda x: len(x) > 0)
    match bool_mapped:
        case Ok(value):
            is_bool(a=value)
            assert value is True
        case Err(_):
            pytest.fail("Should be Ok")

    # int -> list[int]
    int_result2: Result[int, str] = Ok(3)
    list_mapped: Result[list[int], str] = int_result2.map(lambda x: list(range(x)))
    match list_mapped:
        case Ok(value):
            is_list(value)
            assert value == [0, 1, 2]
        case Err(_):
            pytest.fail("Should be Ok")

    # Complex type transformation: str -> dict[str, int]
    str_input: Result[str, bool] = Ok("hello,world")
    dict_mapped: Result[dict[str, int], bool] = str_input.map(
        lambda s: {word: len(word) for word in s.split(",")}
    )
    match dict_mapped:
        case Ok(value):
            assert isinstance(value, dict)
            assert value == {"hello": 5, "world": 5}
        case Err(_):
            pytest.fail("Should be Ok")


def test_err_map_type_unchanged() -> None:
    """Test that map on Err preserves error type but allows Ok type change."""
    err_result: Result[int, str] = Err("error")
    # Even though we're mapping int->str, the error remains unchanged
    mapped: Result[str, str] = err_result.map(lambda x: str(x))
    match mapped:
        case Ok(_):
            pytest.fail("Should be Err")
        case Err(value):
            is_string(value)
            assert value == "error"


def test_ok_map_err() -> None:
    """Test Ok map_err method preserves Ok value."""
    result = Ok(42)
    mapped = result.map_err(lambda e: e.upper())
    assert mapped == Ok(42)


def test_err_map_err() -> None:
    """Test Err map_err method."""
    result = Err("error")
    mapped = result.map_err(lambda e: e.upper())
    assert mapped == Err("ERROR")


def test_ok_map_err_type_unchanged() -> None:
    """Test that map_err on Ok preserves Ok type but allows Err type change."""
    ok_result: Result[int, str] = Ok(42)
    # Even though we're mapping str->int, the Ok value remains unchanged
    mapped: Result[int, int] = ok_result.map_err(lambda e: len(e))
    match mapped:
        case Ok(value):
            is_int(value)
            assert value == 42
        case Err(_):
            pytest.fail("Should be Ok")


def test_err_map_err_type_conversion() -> None:
    """Test that map_err properly converts error types."""
    # str -> int
    str_err_result: Result[int, str] = Err("error")
    int_mapped: Result[int, int] = str_err_result.map_err(lambda e: len(e))
    match int_mapped:
        case Ok(_):
            pytest.fail("Should be Err")
        case Err(value):
            is_int(value)
            assert value == 5

    # str -> bool
    str_err_result2: Result[float, str] = Err("test")
    bool_mapped: Result[float, bool] = str_err_result2.map_err(lambda e: e.startswith("t"))
    match bool_mapped:
        case Ok(_):
            pytest.fail("Should be Err")
        case Err(value):
            is_bool(a=value)
            assert value is True

    # Complex error transformation: str -> dict[str, int]
    complex_err: Result[int, str] = Err("error,message")
    dict_err: Result[int, dict[str, int]] = complex_err.map_err(
        lambda e: {word: len(word) for word in e.split(",")}
    )
    match dict_err:
        case Ok(_):
            pytest.fail("Should be Err")
        case Err(value):
            assert isinstance(value, dict)
            assert value == {"error": 5, "message": 7}


def test_map_chaining() -> None:
    """Test chaining map operations."""
    result = Ok(5)
    chained = result.map(lambda x: x * 2).map(lambda x: x + 1)
    assert chained == Ok(11)


def test_map_err_chaining() -> None:
    """Test chaining map_err operations with type conversion."""

    def foo() -> Result[str, str]:
        return Err("error message")

    # Chain multiple type conversions: str->int for Ok, str->float for Err
    chained_result: Result[int, float] = (
        foo().map(lambda x: len(x)).map_err(lambda e: float(len(e)))
    )
    match chained_result:
        case Ok(data):
            is_int(data)
        case Err(err):
            is_float(err)
            assert err == 13.0  # len("error message")


def test_map_chaining_multiple_types() -> None:
    """Test chaining multiple map operations with different types."""
    result: Result[int, str] = Ok(5)

    # int -> str -> bool -> int
    chained: Result[int, str] = (
        result.map(lambda x: str(x * 2))  # int -> str
        .map(lambda x: x.startswith("1"))  # str -> bool
        .map(lambda x: 1 if x else 0)  # bool -> int
    )

    match chained:
        case Ok(value):
            is_int(value)
            assert value == 1  # "10".startswith("1") is True, so 1
        case Err(_):
            pytest.fail("Should be Ok")


# map_or and map_or_else tests with type conversion
def test_ok_map_or() -> None:
    """Test Ok map_or method."""
    result = Ok(5)
    mapped = result.map_or(0, lambda x: x * 2)
    assert mapped == 10


def test_err_map_or() -> None:
    """Test Err map_or method returns default."""
    result = Err("error")
    mapped = result.map_or(0, lambda x: x * 2)
    assert mapped == 0


def test_ok_map_or_type_conversion() -> None:
    """Test map_or with type conversion."""
    int_result: Result[int, str] = Ok(42)

    # Convert int -> str, with str default
    str_mapped: str = int_result.map_or("default", lambda x: f"value_{x}")
    is_string(str_mapped)
    assert str_mapped == "value_42"

    # Convert int -> bool, with bool default
    bool_mapped: bool = int_result.map_or(default=False, op=lambda x: x > 0)
    is_bool(a=bool_mapped)
    assert bool_mapped is True

    # Convert int -> list[str], with list[str] default
    list_mapped: list[str] = int_result.map_or(["default"], lambda x: [f"item_{x}"])
    is_list(list_mapped)
    assert list_mapped == ["item_42"]


def test_err_map_or_type_conversion() -> None:
    """Test map_or returns default when Err."""
    err_result: Result[int, str] = Err("error")

    # Even though mapping would convert int->str, we get the default
    str_mapped: str = err_result.map_or("default", lambda x: f"value_{x}")
    is_string(str_mapped)
    assert str_mapped == "default"

    # Test with complex types
    complex_mapped: dict[str, int] = err_result.map_or({"default": 0}, lambda x: {"value": x})
    assert isinstance(complex_mapped, dict)
    assert complex_mapped == {"default": 0}


def test_ok_map_or_else() -> None:
    """Test Ok map_or_else method."""
    result = Ok(5)
    mapped = result.map_or_else(lambda: 0, lambda x: x * 2)
    assert mapped == 10


def test_err_map_or_else() -> None:
    """Test Err map_or_else method calls default function."""
    result = Err("error")
    mapped = result.map_or_else(lambda: 42, lambda x: x * 2)
    assert mapped == 42


def test_ok_map_or_else_type_conversion() -> None:
    """Test map_or_else with type conversion."""
    int_result: Result[int, str] = Ok(42)

    # Convert int -> str, with function providing str default
    str_mapped: str = int_result.map_or_else(lambda: "default_string", lambda x: f"converted_{x}")
    is_string(str_mapped)
    assert str_mapped == "converted_42"

    # Convert int -> dict, with function providing dict default
    dict_mapped: dict[str, int] = int_result.map_or_else(
        lambda: {"default": 0}, lambda x: {"value": x}
    )
    assert isinstance(dict_mapped, dict)
    assert dict_mapped == {"value": 42}


def test_err_map_or_else_type_conversion() -> None:
    """Test map_or_else calls default function when Err."""
    err_result: Result[int, str] = Err("error")

    # Default function is called, mapping function is not
    str_mapped: str = err_result.map_or_else(
        lambda: "from_default_func", lambda x: f"converted_{x}"
    )
    is_string(str_mapped)
    assert str_mapped == "from_default_func"


# and_then tests with comprehensive type conversion verification
def test_ok_and_then() -> None:
    """Test Ok and_then method."""
    res = get_result(ok=True)

    def and_then_func(data: str) -> Result[int, str]:
        return Ok(len(data))

    result = res.and_then(and_then_func)
    match result:
        case Ok(data):
            is_int(data)
            assert data == 7  # len("success")
        case Err(_):
            pytest.fail("Should not be Err")


def test_err_and_then() -> None:
    """Test Err and_then method preserves error."""
    res = get_result(ok=False)

    def and_then_func(data: str) -> Result[int, str]:
        return Ok(len(data))

    result = res.and_then(and_then_func)
    match result:
        case Ok(_):
            pytest.fail("Should not be Ok")
        case Err(err):
            assert err == "error"


def test_and_then_type_conversion() -> None:
    """Test and_then with comprehensive type conversions."""
    # str -> int -> bool -> list[int]
    str_result: Result[str, str] = Ok("hello")

    def str_to_int(s: str) -> Result[int, str]:
        try:
            return Ok(len(s))
        except ValueError as e:
            return Err(str(e))

    def int_to_bool(i: int) -> Result[bool, str]:
        return Ok(i > 3)

    def bool_to_list(b: bool) -> Result[list[int], str]:  # noqa: FBT001
        if b:
            return Ok([1, 2, 3])
        return Err("bool was False")

    # Chain the conversions
    final_result: Result[list[int], str] = (
        str_result.and_then(str_to_int).and_then(int_to_bool).and_then(bool_to_list)
    )

    match final_result:
        case Ok(value):
            is_list(value)
            assert value == [1, 2, 3]
        case Err(_):
            pytest.fail("Should be Ok")


def test_and_then_error_type_preservation() -> None:
    """Test that and_then preserves error type through chain."""
    int_result: Result[int, str] = Ok(42)

    def int_to_str_result(i: int) -> Result[str, str]:
        return Ok(str(i))

    def str_to_bool_result(s: str) -> Result[bool, str]:
        return Err("conversion failed")  # Force an error

    # Chain operations - error should be preserved as str
    final_result: Result[bool, str] = int_result.and_then(int_to_str_result).and_then(
        str_to_bool_result
    )

    match final_result:
        case Ok(_):
            pytest.fail("Should be Err")
        case Err(error):
            is_string(error)
            assert error == "conversion failed"


def test_and_then_mixed_error_types() -> None:
    """Test and_then with functions that preserve error type."""
    str_result: Result[str, str] = Ok("42")

    def parse_int(s: str) -> Result[int, str]:  # Same error type
        try:
            return Ok(int(s))
        except ValueError:
            return Err("parse_error")

    # and_then preserves the error type
    int_result: Result[int, str] = str_result.and_then(parse_int)

    match int_result:
        case Ok(value):
            is_int(value)
            assert value == 42
        case Err(_):
            pytest.fail("Should be Ok")


def test_and_then_complex_type_conversions() -> None:
    """Test and_then with complex type conversions."""
    initial: Result[str, int] = Ok("key1,key2,key3")

    def str_to_list(s: str) -> Result[list[str], int]:
        return Ok(s.split(","))

    def list_to_dict(lst: list[str]) -> Result[dict[str, int], int]:
        return Ok({item: len(item) for item in lst})

    def dict_to_tuple(d: dict[str, int]) -> Result[tuple[str, ...], int]:
        return Ok(tuple(d.keys()))

    result: Result[tuple[str, ...], int] = (
        initial.and_then(str_to_list).and_then(list_to_dict).and_then(dict_to_tuple)
    )

    match result:
        case Ok(value):
            assert isinstance(value, tuple)
            assert value == ("key1", "key2", "key3")
        case Err(_):
            pytest.fail("Should be Ok")


def test_and_then_chaining() -> None:
    """Test and_then method chaining."""
    result = Ok(5)
    chained = (
        result.and_then(lambda x: Ok(x * 2))
        .and_then(lambda x: Ok(str(x)))
        .and_then(lambda x: Ok(x + "!"))
    )
    assert chained == Ok("10!")


def test_and_then_early_error() -> None:
    """Test and_then with early error termination."""
    result = Ok(5)
    chained = (
        result.and_then(lambda x: Ok(x * 2))
        .and_then(lambda _: Err("failed"))
        .and_then(lambda x: Ok(x + "!"))
    )
    assert chained == Err("failed")


# or_else tests with comprehensive type conversion verification
def test_ok_or_else() -> None:
    """Test Ok or_else method preserves Ok value."""
    result = Ok(42)
    recovered = result.or_else(lambda _: Ok(0))
    assert recovered == Ok(42)


def test_err_or_else() -> None:
    """Test Err or_else method with recovery."""
    result = Err("error")
    recovered = result.or_else(lambda _: Ok("recovered"))
    assert recovered == Ok("recovered")


def test_err_or_else_still_error() -> None:
    """Test Err or_else method when recovery also fails."""
    result = Err("error1")
    recovered = result.or_else(lambda _: Err("error2"))
    assert recovered == Err("error2")


def test_or_else_type_conversion() -> None:
    """Test or_else with error type conversion."""
    # str error -> int error
    str_err_result: Result[int, str] = Err("string_error")

    def str_err_to_int_err(e: str) -> Result[int, int]:
        return Err(len(e))  # Convert str error to int error

    int_err_result: Result[int, int] = str_err_result.or_else(str_err_to_int_err)

    match int_err_result:
        case Ok(_):
            pytest.fail("Should be Err")
        case Err(error):
            is_int(error)
            assert error == 12  # len("string_error")


def test_or_else_error_recovery_with_type_change() -> None:
    """Test or_else recovering from error with type change."""
    # Start with Result[str, str]
    str_result: Result[str, str] = Err("parse_error")

    def recover_with_int(e: str) -> Result[str, int]:
        # Recover by providing a default str value, but change error type to int
        if "parse" in e:
            return Ok("default_value")
        return Err(404)  # Different error type

    recovered_result: Result[str, int] = str_result.or_else(recover_with_int)

    match recovered_result:
        case Ok(value):
            is_string(value)
            assert value == "default_value"
        case Err(_):
            pytest.fail("Should be Ok")


def test_or_else_chain_multiple_recoveries() -> None:
    """Test chaining multiple or_else operations with type changes."""
    initial_result: Result[int, str] = Err("initial_error")

    def first_recovery(e: str) -> Result[int, int]:
        return Err(len(e))  # str -> int error

    def second_recovery(e: int) -> Result[int, bool]:
        return Err(e > 10)  # int -> bool error

    def final_recovery(e: bool) -> Result[int, float]:  # noqa: FBT001
        if e:
            return Ok(999)  # Successful recovery
        return Err(3.14)  # float error

    final_result: Result[int, float] = (
        initial_result.or_else(first_recovery).or_else(second_recovery).or_else(final_recovery)
    )

    match final_result:
        case Ok(value):
            is_int(value)
            assert value == 999
        case Err(_):
            pytest.fail("Should be Ok after recovery")


def test_or_else_complex_type_conversions() -> None:
    """Test or_else with complex type conversions."""
    initial: Result[dict[str, int], str] = Err("failed_to_parse")

    def str_to_list_err(e: str) -> Result[dict[str, int], list[str]]:
        return Err(e.split("_"))

    def list_to_dict_err(e: list[str]) -> Result[dict[str, int], dict[str, bool]]:
        return Err({item: item.startswith("f") for item in e})

    def dict_to_success(e: dict[str, bool]) -> Result[dict[str, int], tuple[str, ...]]:
        # Finally recover with success
        return Ok({"recovered": 1, "successfully": 2})

    result: Result[dict[str, int], tuple[str, ...]] = (
        initial.or_else(str_to_list_err).or_else(list_to_dict_err).or_else(dict_to_success)
    )

    match result:
        case Ok(value):
            assert isinstance(value, dict)
            assert value == {"recovered": 1, "successfully": 2}
        case Err(_):
            pytest.fail("Should be Ok after recovery")


# inspect tests
def test_ok_inspect() -> None:
    """Test Ok inspect method calls function and returns original."""
    called_with = []
    result = Ok(42)

    def inspector(value: int) -> None:
        called_with.append(value)

    returned = result.inspect(inspector)
    assert returned == Ok(42)
    assert called_with == [42]


def test_err_inspect() -> None:
    """Test Err inspect method does not call function."""
    called_with = []
    result = Err("error")

    def inspector(value: str) -> None:
        called_with.append(value)

    returned = result.inspect(inspector)
    assert returned == Err("error")
    assert called_with == []  # Should not be called for Err


def test_ok_inspect_err() -> None:
    """Test Ok inspect_err method does not call function."""
    called_with = []
    result = Ok(42)

    def inspector(value: int) -> None:
        called_with.append(value)

    returned = result.inspect_err(inspector)
    assert returned == Ok(42)
    assert called_with == []  # Should not be called for Ok


def test_err_inspect_err() -> None:
    """Test Err inspect_err method calls function and returns original."""
    called_with = []
    result = Err("error")

    def inspector(value: str) -> None:
        called_with.append(value)

    returned = result.inspect_err(inspector)
    assert returned == Err("error")
    assert called_with == ["error"]


# Equality and hashing tests
def test_ok_equality() -> None:
    """Test Ok equality and hashing."""
    ok1 = Ok(42)
    ok2 = Ok(42)
    ok3 = Ok(43)
    err1 = Err(42)

    assert ok1 == ok2
    assert ok1 != ok3
    assert ok1 != err1
    assert hash(ok1) == hash(ok2)
    assert hash(ok1) != hash(ok3)


def test_err_equality() -> None:
    """Test Err equality and hashing."""
    err1 = Err("error")
    err2 = Err("error")
    err3 = Err("other")
    ok1 = Ok("error")

    assert err1 == err2
    assert err1 != err3
    assert err1 != ok1
    assert hash(err1) == hash(err2)
    assert hash(err1) != hash(err3)


# Type guard tests
def test_is_ok_type_guard() -> None:
    """Test is_ok type guard function."""
    result: Result[int, str] = Ok(42)
    assert is_ok(result)
    assert not is_err(result)


def test_is_err_type_guard() -> None:
    """Test is_err type guard function."""
    result: Result[int, str] = Err("error")
    assert is_err(result)
    assert not is_ok(result)


# Real-world use case tests
def divide(a: int, b: int) -> Result[float, str]:
    """Divide two numbers, returning an error for division by zero."""
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)


def test_division_success() -> None:
    """Test successful division operation."""
    result = divide(10, 2)
    assert result == Ok(5.0)
    assert result.unwrap() == 5.0


def test_division_failure() -> None:
    """Test division by zero operation."""
    result = divide(10, 0)
    assert result == Err("Division by zero")
    assert result.unwrap_or(0.0) == 0.0


def parse_int(s: str) -> Result[int, str]:
    """Parse a string into an integer."""
    try:
        return Ok(int(s))
    except ValueError:
        return Err(f"Could not parse '{s}' as integer")


def test_parse_int_success() -> None:
    """Test successful integer parsing."""
    result = parse_int("42")
    assert result == Ok(42)


def test_parse_int_failure() -> None:
    """Test failed integer parsing."""
    result = parse_int("not_a_number")
    assert result.is_err()
    assert "Could not parse" in result.unwrap_err()


def test_chained_operations() -> None:
    """Test a realistic scenario with multiple operations."""

    def safe_sqrt(x: float) -> Result[float, str]:
        if x < 0:
            return Err("Cannot take square root of negative number")
        return Ok(x**0.5)

    def format_result(x: float) -> Result[str, str]:
        return Ok(f"{x:.2f}")

    # Success case
    result = parse_int("16").map(float).and_then(safe_sqrt).and_then(format_result)

    assert result == Ok("4.00")

    # Failure case - bad input
    result = parse_int("not_a_number").map(float).and_then(safe_sqrt).and_then(format_result)

    assert result.is_err()
    assert "Could not parse" in result.unwrap_err()

    # Failure case - negative number
    result = parse_int("-16").map(float).and_then(safe_sqrt).and_then(format_result)

    assert result.is_err()
    assert "Cannot take square root" in result.unwrap_err()


def test_result_with_optional() -> None:
    """Test interoperability with Optional types."""

    def to_optional(result: Result[int, str]) -> int | None:
        return result.ok()

    def from_optional(opt: int | None, error: str) -> Result[int, str]:
        return Ok(opt) if opt is not None else Err(error)

    ok_result = Ok(42)
    err_result = Err("error")

    assert to_optional(ok_result) == 42
    assert to_optional(err_result) is None

    assert from_optional(42, "failed") == Ok(42)
    assert from_optional(None, "failed") == Err("failed")


def test_error_accumulation() -> None:
    """Test collecting multiple errors."""

    def validate_positive(x: int) -> Result[int, str]:
        return Ok(x) if x > 0 else Err(f"{x} is not positive")

    def validate_even(x: int) -> Result[int, str]:
        return Ok(x) if x % 2 == 0 else Err(f"{x} is not even")

    def validate_number(x: int) -> Result[int, list[str]]:
        errors = []

        pos_result = validate_positive(x)
        if pos_result.is_err():
            errors.append(pos_result.unwrap_err())

        even_result = validate_even(x)
        if even_result.is_err():
            errors.append(even_result.unwrap_err())

        return Err(errors) if errors else Ok(x)

    assert validate_number(4) == Ok(4)  # positive and even
    assert validate_number(-2) == Err(["-2 is not positive"])  # even but not positive
    assert validate_number(3) == Err(["3 is not even"])  # positive but not even
    assert validate_number(-3) == Err(["-3 is not positive", "-3 is not even"])  # neither


def test_unwrap_error_details() -> None:
    """Test UnwrapError exception details."""
    ok_result = Ok(42)
    err_result = Err("error message")

    # Test that UnwrapError preserves the original result
    with pytest.raises(UnwrapError) as exc_info:
        err_result.unwrap()
    assert exc_info.value.result == err_result
    assert str(exc_info.value) == "Called `Result.unwrap()` on an `Err` value"

    with pytest.raises(UnwrapError) as exc_info:
        ok_result.unwrap_err()
    assert exc_info.value.result == ok_result
    assert str(exc_info.value) == "Called `Result.unwrap_err()` on an `Ok` value"

    with pytest.raises(UnwrapError) as exc_info:
        err_result.expect("Custom error message")
    assert exc_info.value.result == err_result
    assert str(exc_info.value) == "Custom error message"


# Edge cases and corner cases
def test_none_values() -> None:
    """Test handling None values."""
    ok_none = Ok(None)
    err_none = Err(None)

    assert ok_none.is_ok()
    assert ok_none.unwrap() is None
    assert ok_none.ok() is None

    assert err_none.is_err()
    assert err_none.unwrap_err() is None
    assert err_none.err() is None


def test_nested_results() -> None:
    """Test Results containing other Results."""
    nested_ok = Ok(Ok(42))
    nested_err = Ok(Err("inner error"))

    assert nested_ok.unwrap() == Ok(42)
    assert nested_err.unwrap() == Err("inner error")

    # Flatten nested Ok
    flattened = nested_ok.and_then(lambda x: x)
    assert flattened == Ok(42)

    # Flatten nested Err
    flattened = nested_err.and_then(lambda x: x)
    assert flattened == Err("inner error")


def test_large_values() -> None:
    """Test with large values to ensure no issues with memory/performance."""
    large_list = list(range(10000))
    result = Ok(large_list)

    assert result.is_ok()
    assert len(result.unwrap()) == 10000

    mapped = result.map(len)
    assert mapped == Ok(10000)


# Additional comprehensive type conversion tests
def test_complex_type_conversion_pipeline() -> None:
    """Test a complex pipeline with multiple type conversions."""

    def parse_number(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"Cannot parse '{s}' as int")

    def validate_positive(n: int) -> Result[int, str]:
        return Ok(n) if n > 0 else Err(f"{n} is not positive")

    def square_root(n: int) -> Result[float, str]:
        # Math import is at the top of the file

        return Ok(math.sqrt(n))

    def format_result(f: float) -> Result[str, str]:
        return Ok(f"{f:.3f}")

    # Test successful pipeline: str -> int -> int -> float -> str
    input_result: Result[str, str] = Ok("16")

    final_result: Result[str, str] = (
        input_result.and_then(parse_number)
        .and_then(validate_positive)
        .and_then(square_root)
        .and_then(format_result)
    )

    match final_result:
        case Ok(value):
            is_string(value)
            assert value == "4.000"
        case Err(_):
            pytest.fail("Should be Ok")

    # Test failure in middle of pipeline
    input_result2: Result[str, str] = Ok("-16")

    final_result2: Result[str, str] = (
        input_result2.and_then(parse_number)
        .and_then(validate_positive)  # This will fail
        .and_then(square_root)
        .and_then(format_result)
    )

    match final_result2:
        case Ok(_):
            pytest.fail("Should be Err")
        case Err(error):
            is_string(error)
            assert "-16 is not positive" in error


def test_all_type_convertors_preserve_types() -> None:
    """Comprehensive test that all type convertors work with pyright."""
    # Test map with type conversion
    int_ok: Result[int, str] = Ok(42)
    str_result: Result[str, str] = int_ok.map(lambda x: f"num_{x}")

    match str_result:
        case Ok(s):
            is_string(s)
        case Err(e):
            is_string(e)

    # Test map_err with type conversion
    str_err: Result[int, str] = Err("error")
    int_err_result: Result[int, int] = str_err.map_err(lambda e: len(e))

    match int_err_result:
        case Ok(i):
            is_int(i)
        case Err(i):
            is_int(i)

    # Test and_then with type conversion
    str_ok: Result[str, int] = Ok("hello")
    bool_result: Result[bool, int] = str_ok.and_then(lambda s: Ok(len(s) > 3))

    match bool_result:
        case Ok(b):
            is_bool(a=b)
        case Err(i):
            is_int(i)

    # Test or_else with type conversion
    int_err: Result[str, int] = Err(404)
    str_err_result: Result[str, str] = int_err.or_else(lambda i: Err(f"Error code: {i}"))

    match str_err_result:
        case Ok(s):
            is_string(s)
        case Err(s):
            is_string(s)


def test_type_convertor_complex_callbacks() -> None:
    """Test type convertors with more complex typed callbacks."""
    # Test map with callback that changes generic types
    list_result: Result[list[int], str] = Ok([1, 2, 3, 4, 5])

    def list_to_dict(items: list[int]) -> dict[int, str]:
        return {i: f"item_{i}" for i in items}

    dict_result: Result[dict[int, str], str] = list_result.map(list_to_dict)

    match dict_result:
        case Ok(d):
            assert isinstance(d, dict)
            assert d == {1: "item_1", 2: "item_2", 3: "item_3", 4: "item_4", 5: "item_5"}
        case Err(_):
            pytest.fail("Should be Ok")

    # Test and_then with callback returning different generic types
    def dict_to_list_result(d: dict[int, str]) -> Result[list[str], str]:
        return Ok(list(d.values()))

    list_result2: Result[list[str], str] = dict_result.and_then(dict_to_list_result)

    match list_result2:
        case Ok(lst):
            is_list(lst)
            assert all(isinstance(item, str) for item in lst)
        case Err(s):
            is_string(s)

    # Test map_or with complex types
    err_dict: Result[dict[str, int], float] = Err(3.14)
    default_tuple: tuple[str, ...] = ("default", "values")

    tuple_result: tuple[str, ...] = err_dict.map_or(default_tuple, lambda d: tuple(d.keys()))

    assert isinstance(tuple_result, tuple)
    assert tuple_result == ("default", "values")


def test_async_type_annotations() -> None:
    """Test that async methods would work with proper types (syntax check only)."""
    # These tests just verify the type annotations compile correctly
    # We don't actually run async code in these tests

    async def async_transform(x: int) -> str:
        return f"async_{x}"

    async def async_result_transform(x: int) -> Result[str, bool]:
        return Ok(f"async_result_{x}")

    # These functions demonstrate correct async typing but are not actually called


def test_generic_type_preservation() -> None:
    """Test that generic types are properly preserved through transformations."""
    # Test with custom generic classes
    # Generic and TypeVar imports are at the top of the file

    T = TypeVar("T")

    class Container(Generic[T]):
        def __init__(self, value: T) -> None:
            self.value = value

        def get(self) -> T:
            return self.value

    # Test map preserves generic structure
    container_result: Result[Container[int], str] = Ok(Container(42))

    def extract_from_container(c: Container[int]) -> int:
        return c.get()

    int_result: Result[int, str] = container_result.map(extract_from_container)

    match int_result:
        case Ok(value):
            is_int(value)
            assert value == 42
        case Err(_):
            pytest.fail("Should be Ok")

    # Test and_then with generic preservation
    def int_to_container_result(i: int) -> Result[Container[str], str]:
        return Ok(Container(str(i)))

    str_container_result: Result[Container[str], str] = int_result.and_then(int_to_container_result)

    match str_container_result:
        case Ok(container):
            assert isinstance(container, Container)
            is_string(container.get())
            assert container.get() == "42"
        case Err(_):
            pytest.fail("Should be Ok")
