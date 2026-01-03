# Python Code Standards

## Core Philosophy

Write minimalistic, purposeful code. Every line must earn its place. No over-engineering, no unnecessary abstractions, no "just in case" features. Build exactly what's asked for, nothing more.

## Variable Naming

- Use full, descriptive names: `user_authentication_token` not `auth_tok` or `token`
- Booleans start with `is_`, `has_`, `should_`: `is_valid`, `has_permission`, `should_retry`
- Functions/methods are verbs: `calculate_total()`, `fetch_user_data()`, `validate_input()`
- Constants are UPPER_CASE: `MAX_RETRY_ATTEMPTS`, `DEFAULT_TIMEOUT`
- Avoid single letters except for: `i`, `j`, `k` in simple loops, `e` for exceptions, `f` for files
- No abbreviations unless universally known (`id`, `url`, `http`, `api`)

## Function Design

- One function = one purpose (single responsibility)
- Keep functions under 20 lines when possible
- If a function does multiple things, split it
- Function names must clearly indicate what they do
- Use type hints always:
  ```python
  def calculate_discount(price: float, discount_rate: float) -> float:
      """Calculate final price after discount."""
      return price * (1 - discount_rate)
  ```

## Documentation

- Use Google-style docstrings consistently
- Every function gets a docstring (even if it seems obvious):

  ```python
  def validate_email(email: str) -> bool:
      """
      Check if email address is valid format.

      Args:
          email: Email address to validate

      Returns:
          True if valid format, False otherwise

      Raises:
          ValueError: If email is empty string
      """
  ```

- Docstrings explain WHAT and WHY, not HOW (code shows how)
- Add inline comments only for non-obvious logic or business rules
- No redundant comments: `# increment counter` is useless

## Code Organization

- Imports at top, grouped: stdlib, third-party, local
- One blank line between functions, two between classes
- Maximum line length: 125 characters (matches Ruff config)
- Group related functionality together

## Project Structure

- Source code in `src/`
- Tests in `tests/`
- Data files in `data/`
- Models/artifacts in `models/`
- One module = one responsibility
- Shared utilities belong in `utils.py` or a dedicated module
- Keep related functionality together in the same file

## Constants & Configuration

- Define constants in a central location (`utils.py` or `config.py`)
- Never duplicate magic numbers or strings across files
- Use ALL_CAPS naming for constants: `MAX_RETRY_ATTEMPTS`, `API_BASE_URL`
- Group related constants together with comments
- Configuration that varies by environment goes in environment variables

## Dependency Management

- Minimize external dependencies - only add what's truly needed
- Pin versions in `pyproject.toml` for reproducibility
- Document why non-obvious dependencies are needed
- Prefer well-maintained, widely-used libraries
- Check license compatibility before adding dependencies

## Linting & Formatting

- Use Ruff for linting and formatting (configured for 125 char line width)
- Code must pass `ruff check` with no errors
- Run `ruff format` before committing
- Fix all Ruff warnings - don't ignore them without good reason

## Error Handling

- Be specific with exceptions: catch `FileNotFoundError`, not `Exception`
- Always handle potential failures (file operations, network calls, user input)
- Fail fast: validate inputs early, at function start
- Use `logging` module, not print statements for errors:

  ```python
  import logging

  logging.error(f"Failed to process user {user_id}: {str(e)}")
  ```

## Security Best Practices

- Never hardcode credentials, API keys, or secrets
- Use environment variables: `os.getenv('API_KEY')`
- Validate and sanitize ALL user input
- Use parameterized queries, never string concatenation for SQL
- Don't log sensitive data (passwords, tokens, personal info)
- Use `secrets` module for tokens, not `random`

## Performance

- Use appropriate data structures: `set` for membership tests, `dict` for lookups
- Avoid nested loops when possible
- Use list comprehensions for simple transformations, but stay readable
- Don't optimize prematurely - write clear code first, profile if needed
- Close resources properly: use context managers (`with` statements)

## What to Avoid

- No print statements in production code (use logging)
- No commented-out code (delete it, that's what git is for)
- No complex one-liners that sacrifice readability
- No global variables (pass data explicitly)
- No bare `except:` clauses (always specify exception type)
- No mutable default arguments: `def func(items: list = None)` not `def func(items=[])`

## Debug Output

- Use logging with appropriate levels
- Every log message must start with an emoji for visual clarity:

  ```python
  logger.debug("🔍 Processing started")    # Detailed diagnostic
  logger.info("✅ User logged in")         # General info / success
  logger.warning("⚠️ Retry attempt 3")     # Something unexpected
  logger.error("❌ Database unreachable")  # Error occurred
  ```

- Standard emoji conventions:
  - ✅ Success, completed, loaded
  - ❌ Error, failure
  - ⚠️ Warning, caution
  - 📂 File/data loading
  - 💾 Saving files
  - 🔧 Processing, configuration
  - 📊 Statistics, data summaries
  - 🔍 Searching, detecting
  - 📅 Time, dates, scheduling
  - 🤖 Model operations
  - 🎯 Accuracy, targets
  - ⏭️ Skipping
  - 🚀 Launch, start
- Include context in logs: IDs, timestamps, relevant values
- Remove or comment out debug logs before committing

## Code Style

- Follow PEP 8 conventions (enforced by Ruff)
- Use f-strings for formatting: `f"Hello {name}"` not `"Hello " + name`
- Prefer explicit over implicit: `if value is not None:` not `if value:`
- Use guard clauses to reduce nesting:

  ```python
  # Good
  if not user:
      return None

  # Process user...

  # Bad
  if user:
      # Process user (deeply nested)
  ```

## Testing

- Test files mirror source structure: `src/foo.py` → `tests/test_foo.py`
- Test function naming: `test_<function>_<scenario>_<expected_result>`

  ```python
  def test_calculate_points_first_place_returns_25():
      assert calculate_points(1) == 25

  def test_calculate_points_outside_top_10_returns_0():
      assert calculate_points(11) == 0
  ```

- Use pytest fixtures for shared setup
- Test edge cases, not just happy paths
- Write functions that are easy to test (pure functions when possible)
- Separate I/O from logic
- Make dependencies explicit (pass them in, don't hide them)
- Mock external dependencies (APIs, databases, file system)

## Remember

Clean code is code that:

1. Does exactly what it should (no more, no less)
2. Is obvious to read 6 months from now
3. Is easy to modify when requirements change
4. Handles errors gracefully
5. Fails fast when something is wrong

If code requires explanation beyond a brief comment, it's probably too complex. Simplify.
