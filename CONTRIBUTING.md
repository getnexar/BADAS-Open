# Contributing to BADAS-Open

First off, thank you for considering contributing to BADAS-Open! It's people like you that make BADAS-Open such a great tool for the autonomous driving community.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and explain why it's a problem**
- **Include screenshots and animated GIFs if possible**
- **Include your environment details** (OS, Python version, PyTorch version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain why it's insufficient**
- **Explain why this enhancement would be useful**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Write clear, commented code** following our style guidelines
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** before submitting
6. **Write a clear PR description** explaining your changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/badas-open.git
cd badas-open

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Style Guidelines

### Python Style

We use [Black](https://github.com/psf/black) for code formatting and [isort](https://github.com/PyCQA/isort) for import sorting:

```bash
# Format code
black badas/
isort badas/

# Check style
flake8 badas/
mypy badas/
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add sliding window inference for long videos

- Implement chunked video processing
- Add overlap parameter for smooth predictions
- Optimize memory usage for edge devices

Fixes #123
```

### Documentation

- Use docstrings for all public functions, classes, and modules
- Follow [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings
- Update README.md if adding new features
- Add examples for new functionality

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=badas tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions

## Project Structure

```
badas-open/
├── badas/              # Main package code
│   ├── core/          # Core functionality
│   ├── models/        # Model implementations
│   └── utils/         # Utility functions
├── tests/             # Test files
├── examples/          # Example scripts
├── docs/              # Documentation
└── .github/           # GitHub specific files
    └── workflows/     # CI/CD workflows
```

## Recognition

Contributors will be recognized in the following ways:
- Added to CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- Given credit in academic publications when applicable

## Questions?

Feel free to open an issue with the label "question" or reach out to the maintainers at research@nexar.com.

Thank you for contributing to BADAS-Open! 🚗💨