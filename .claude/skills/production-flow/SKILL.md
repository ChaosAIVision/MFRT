# Production Flow - Backend Production Ready Development

Backend production-ready development workflow with comprehensive testing and reporting.

## Core Principles

### Code Style & Quality
- Write READABLE code that is easy to maintain and understand
- Use proper logging with Python's logging module (NEVER use print())
- Configure logging with appropriate levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Log errors with full context: error message, stack trace, input data
- Use meaningful variable names and function names
- Add docstrings for all classes and functions
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values

### Testing Protocol (CRITICAL)

#### 1. Test After Each Component
- Write tests immediately after implementing a component
- DO NOT move to next component until current one is fully tested

#### 2. Test Cases Requirements
- Use REAL test cases, NOT dummy data
- Cover normal cases, edge cases, and error cases
- Test all input/output scenarios
- Test integration with related components

#### 3. When Code Changes
- Re-test the modified component
- Test ALL related/dependent components
- Remove dead code and unused functions
- Update tests if component interface changes

#### 4. Test Failures
- Debug and fix immediately
- Do not proceed until all tests pass
- Document what was broken and how it was fixed

### Component Report (MANDATORY)

After testing each component successfully, create a report in `report_component_task/`:

**Report Filename:** `task_[component_name].md`

**Report Must Include:**

```markdown
# Component: [Component Name]

## Overview
- **Purpose**: What this component does
- **Location**: File path and line numbers
- **Created**: [Date]
- **Status**: ✅ Tested and Working

## Component Details

### Input
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| param1 | str | Yes | [Desc] | "example" | [Rules] |

### Output
**Success Response:**
```json
{"status": "success", "data": {...}}
```

**Error Response:**
```json
{"status": "error", "error": {...}}
```

### Dependencies
- Component A: why needed
- Component B: why needed
- External library: version, purpose

## How It Works
1. Step-by-step logic flow
2. Key algorithms or patterns used
3. Error handling strategy

## Testing Results

### Test Cases
1. **Test Case 1**: [Description]
   - Input: [Real data used]
   - Expected: [Expected result]
   - Actual: [Actual result]
   - Status: ✅ PASS / ❌ FAIL

### Test Coverage
- Line coverage: X%
- Branch coverage: Y%
- Edge cases covered: [List]

### Issues Found & Fixed
1. **Issue**: [Description]
   - **Cause**: [Root cause]
   - **Fix**: [How it was fixed]
   - **Prevention**: [How to prevent in future]

## Production Readiness

### Checklist
- [x] All tests passing
- [x] Logging implemented
- [x] Error handling complete
- [x] Input validation added
- [x] Documentation complete
- [x] No print() statements
- [x] Type hints added
- [x] Dead code removed

### Performance
- Average execution time: X ms
- Memory usage: Y MB
- Bottlenecks identified: [None/List]

### Security
- Input sanitization: Yes/No
- SQL injection protection: Yes/No/N/A
- Authentication required: Yes/No
- Authorization checked: Yes/No/N/A

## Integration

### Used By
- Component X: how it uses this
- Component Y: how it uses this

### Uses
- Component A: what it gets from it
- Component B: what it gets from it

## Code Examples

```python
# Example usage
from module import Component

# Example 1: Normal case
result = Component.method(param1, param2)

# Example 2: Error handling
try:
    result = Component.method(invalid_data)
except ValueError as e:
    logger.error(f"Validation failed: {e}")
```

## Logging Examples

```
[2026-01-28 10:30:45] INFO - Component initialized successfully
[2026-01-28 10:30:46] DEBUG - Processing input: {"key": "value"}
[2026-01-28 10:30:47] WARNING - Rate limit approaching: 90/100
[2026-01-28 10:30:48] ERROR - Database connection failed: timeout after 30s
```

## Next Steps
- [ ] Component X depends on this - implement next
- [ ] Optimize if needed
- [ ] Add caching if performance issue

## Notes
- Additional information
- Lessons learned
- Future improvements
```

## Workflow Steps

When this skill is activated, follow these steps:

### 1. Understand Requirements
- Read task description carefully
- Identify all components needed
- List dependencies between components

### 2. Setup Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

### 3. Implement Component
- Write clean, readable code
- Add type hints
- Add docstrings
- Implement error handling
- Add logging at key points

### 4. Write Tests
- Create test file in `tests/` directory
- Write test cases with REAL data
- Test normal flow
- Test edge cases
- Test error conditions
- Test integration with other components

### 5. Run Tests

```bash
pytest tests/test_component.py -v --cov=src/component
```

### 6. Fix Issues
- If tests fail, debug and fix
- Re-run all related tests
- Remove unused code

### 7. Generate Report
- Create markdown file in `report_component_task/`
- Filename: `task_[component_name].md`
- Fill in all sections of the template
- Include actual test results
- Document any issues and fixes

### 8. Code Review Checklist
- No print() statements (use logging)
- All functions have docstrings
- Type hints on all parameters
- Error handling for all failure points
- Input validation for all external data
- All tests passing
- Code coverage > 80%
- No dead code
- No hardcoded values (use config)

## Important Rules

- NEVER skip testing
- NEVER use dummy/fake test data
- NEVER use print() for debugging (use logging)
- ALWAYS write report after component is tested
- ALWAYS test related components when making changes
- ALWAYS remove unused code
- ALWAYS document issues found during testing
- If you need additional information to create real test cases, DOCUMENT it in the report

## Example: Logging vs Print

❌ **BAD:**
```python
print("User logged in:", user_id)
print("Error:", str(e))
```

✅ **GOOD:**
```python
logger.info(f"User logged in successfully", extra={"user_id": user_id})
logger.error(f"Login failed for user {user_id}", exc_info=True, extra={"email": email})
```

## Example: Real vs Dummy Test Data

❌ **BAD:**
```python
def test_login():
    result = login("fake_user", "fake_pass")
    assert result == "fake_token"
```

✅ **GOOD:**
```python
def test_login():
    # Setup: Create real test user in test DB
    test_user = create_test_user(email="test@example.com", password="Test123!")

    # Execute
    result = login("test@example.com", "Test123!")

    # Verify
    assert result["status"] == "success"
    assert "access_token" in result
    assert jwt.decode(result["access_token"])["user_id"] == test_user.id
```

## Production Readiness Checklist

Before marking component as complete, verify:

- [ ] All tests passing (100%)
- [ ] Code coverage > 80%
- [ ] Logging implemented (no print statements)
- [ ] Error handling complete
- [ ] Input validation complete
- [ ] Type hints added to all functions
- [ ] Docstrings complete
- [ ] Dead code removed
- [ ] Security review done (input sanitization, SQL injection protection)
- [ ] Performance acceptable (< 500ms for most operations)
- [ ] Integration tests with dependent components pass
- [ ] Documentation complete (README, API docs if applicable)
- [ ] Configuration externalized (no hardcoded secrets/URLs)
- [ ] Report generated in `report_component_task/`

## Tools Available

Use these tools during development:
- Read - Read files
- Write - Create new files
- Edit - Modify existing files
- Bash - Run tests, install dependencies
- Glob - Find files by pattern
- Grep - Search code
- TodoWrite - Track tasks
