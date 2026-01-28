#!/bin/bash
# Script to run chaos-auto-prompt unit tests

# Set required environment variables (can use fake values for unit tests)
export OPENAI_API_KEY=test-openai-key-12345
export GOOGLE_API_KEY=test-google-key-67890

# Add source to Python path
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")/../src"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}chaos-auto-prompt Unit Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Parse command line arguments
TEST_FILE=""
VERBOSE=""
COVERAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            TEST_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=chaos_auto_prompt --cov-report=term-missing --cov-report=html"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --file FILE    Run specific test file"
            echo "  -v, --verbose      Verbose output"
            echo "  -c, --coverage     Generate coverage report"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run all tests"
            echo "  $0 -f test_settings.py          # Run specific file"
            echo "  $0 -v -c                        # Run with coverage and verbose output"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="python3 -m pytest tests/unit/"

if [ -n "$TEST_FILE" ]; then
    PYTEST_CMD="python3 -m pytest tests/unit/$TEST_FILE"
fi

PYTEST_CMD="$PYTEST_CMD $VERBOSE $COVERAGE"

# Display command
echo -e "${YELLOW}Running:${NC} $PYTEST_CMD"
echo ""

# Run tests
if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"

    if [ -n "$COVERAGE" ]; then
        echo ""
        echo -e "${YELLOW}Coverage report generated:${NC}"
        echo -e "  Terminal: ${GREEN}Already displayed above${NC}"
        echo -e "  HTML:     ${GREEN}htmlcov/index.html${NC}"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
