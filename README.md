### Result type for Python

```py
from result import Result, Ok, Err


def create_user(name: str, email: str) -> Result[User, AlreadyExistErr]:
	if db.exists(name=name):
		return Err(AlreadyExistErr(name))
	user = User(name, email)
	db.create_user(user)
	return Ok(user)
```

### Caveates

 - `and_then` and `and_then_async` are not the same as their rust equivilant 
and they unify the Error type instead of enforcing you to return the same error type as 
what you had.
The main reason for that was that I needed to migrate from https://github.com/rustedpy/result and I already used that in many places in my code. 

- if you come from https://github.com/rustedpy/result I removed a few features so do note that not everything would work
