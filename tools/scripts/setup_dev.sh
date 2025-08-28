#!/bin/bash

# =============================================================================
# Oasis Crypto Trade - Development Environment Setup Script
# =============================================================================
# Automated setup for development environment
# 
# Author: Oasis Trading Systems
# License: Proprietary
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
MIN_PYTHON_VERSION="3.11.0"
PROJECT_NAME="Oasis Crypto Trade"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${WHITE}                      OASIS CRYPTO TRADE                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${WHITE}                Development Environment Setup                  ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}▶ $1${NC}"
    echo -e "${BLUE}$(printf '%.60s' "$(printf '=%.0s' {1..60})")${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

version_greater_equal() {
    local version1=$1
    local version2=$2
    
    # Convert versions to comparable format
    v1=$(echo "$version1" | awk -F. '{ printf "%d%03d%03d", $1, $2, $3 }')
    v2=$(echo "$version2" | awk -F. '{ printf "%d%03d%03d", $1, $2, $3 }')
    
    [ "$v1" -ge "$v2" ]
}

prompt_yes_no() {
    local question=$1
    local default=${2:-"n"}
    
    if [ "$default" = "y" ]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi
    
    read -p "$(echo -e "${YELLOW}$question $prompt: ${NC}")" answer
    
    if [ -z "$answer" ]; then
        answer=$default
    fi
    
    case "$answer" in
        [Yy]|[Yy][Ee][Ss]) return 0 ;;
        *) return 1 ;;
    esac
}

# =============================================================================
# SYSTEM CHECKS
# =============================================================================

check_operating_system() {
    print_section "System Information"
    
    OS_NAME=$(uname -s)
    OS_VERSION=$(uname -r)
    
    case "$OS_NAME" in
        Linux*)
            DISTRO=""
            if [ -f /etc/os-release ]; then
                DISTRO=$(. /etc/os-release; echo $PRETTY_NAME)
            fi
            print_info "Operating System: Linux ($DISTRO)"
            ;;
        Darwin*)
            MACOS_VERSION=$(sw_vers -productVersion)
            print_info "Operating System: macOS $MACOS_VERSION"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            print_info "Operating System: Windows (Git Bash/WSL)"
            ;;
        *)
            print_warning "Operating System: $OS_NAME (untested)"
            ;;
    esac
    
    print_info "Kernel Version: $OS_VERSION"
    print_info "Architecture: $(uname -m)"
}

check_python() {
    print_section "Python Installation"
    
    if check_command python3; then
        PYTHON_CMD="python3"
    elif check_command python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python $PYTHON_VERSION or higher."
        print_info "Visit: https://www.python.org/downloads/"
        exit 1
    fi
    
    CURRENT_PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_info "Python found: $CURRENT_PYTHON_VERSION"
    
    if version_greater_equal "$CURRENT_PYTHON_VERSION" "$MIN_PYTHON_VERSION"; then
        print_success "Python version is compatible"
    else
        print_error "Python $MIN_PYTHON_VERSION or higher is required"
        print_info "Current version: $CURRENT_PYTHON_VERSION"
        exit 1
    fi
    
    # Check pip
    if $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        PIP_VERSION=$($PYTHON_CMD -m pip --version | awk '{print $2}')
        print_success "pip found: $PIP_VERSION"
    else
        print_error "pip not found. Please install pip."
        exit 1
    fi
}

check_git() {
    print_section "Git Configuration"
    
    if check_command git; then
        GIT_VERSION=$(git --version | awk '{print $3}')
        print_success "Git found: $GIT_VERSION"
        
        # Check git configuration
        if git config user.name >/dev/null && git config user.email >/dev/null; then
            print_success "Git is configured"
            print_info "User: $(git config user.name) <$(git config user.email)>"
        else
            print_warning "Git user not configured"
            if prompt_yes_no "Would you like to configure Git now?"; then
                read -p "Enter your name: " git_name
                read -p "Enter your email: " git_email
                git config --global user.name "$git_name"
                git config --global user.email "$git_email"
                print_success "Git configuration saved"
            fi
        fi
    else
        print_error "Git not found. Please install Git."
        print_info "Visit: https://git-scm.com/downloads"
        exit 1
    fi
}

check_docker() {
    print_section "Docker Installation"
    
    if check_command docker; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_success "Docker found: $DOCKER_VERSION"
        
        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            print_success "Docker daemon is running"
        else
            print_warning "Docker daemon is not running"
            print_info "Please start Docker and run this script again"
        fi
    else
        print_warning "Docker not found"
        print_info "Docker is optional but recommended for development"
        print_info "Visit: https://docs.docker.com/get-docker/"
    fi
    
    if check_command docker-compose; then
        COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
        print_success "Docker Compose found: $COMPOSE_VERSION"
    else
        print_warning "Docker Compose not found"
        print_info "Docker Compose is needed for infrastructure services"
    fi
}

# =============================================================================
# INSTALLATION FUNCTIONS
# =============================================================================

install_poetry() {
    print_section "Poetry Installation"
    
    if check_command poetry; then
        POETRY_VERSION=$(poetry --version | awk '{print $3}')
        print_success "Poetry found: $POETRY_VERSION"
        return 0
    fi
    
    print_info "Installing Poetry..."
    
    if check_command curl; then
        curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    else
        print_error "curl not found. Please install curl first."
        exit 1
    fi
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    if check_command poetry; then
        print_success "Poetry installed successfully"
    else
        print_error "Poetry installation failed"
        print_info "Please install Poetry manually: https://python-poetry.org/docs/#installation"
        exit 1
    fi
}

setup_python_environment() {
    print_section "Python Environment Setup"
    
    cd "$PROJECT_DIR"
    
    # Install dependencies with Poetry
    print_info "Installing project dependencies..."
    poetry install
    
    print_success "Python environment setup complete"
}

setup_pre_commit() {
    print_section "Pre-commit Hooks Setup"
    
    cd "$PROJECT_DIR"
    
    # Install pre-commit hooks
    print_info "Installing pre-commit hooks..."
    poetry run pre-commit install
    
    # Run pre-commit on all files to test
    print_info "Running pre-commit on all files..."
    poetry run pre-commit run --all-files || true
    
    print_success "Pre-commit hooks installed"
}

create_environment_file() {
    print_section "Environment Configuration"
    
    cd "$PROJECT_DIR"
    
    if [ -f ".env" ]; then
        print_warning ".env file already exists"
        if prompt_yes_no "Do you want to overwrite it?"; then
            cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
            print_info "Backup created: .env.backup.$(date +%Y%m%d_%H%M%S)"
        else
            print_info "Keeping existing .env file"
            return 0
        fi
    fi
    
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created from template"
    else
        print_warning ".env.example not found, creating basic .env file"
        cat > .env << EOF
# Basic Oasis Development Configuration
ENVIRONMENT=development
DEBUG=true
TESTING=false

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=oasis_trading_db
POSTGRES_USER=oasis_admin
POSTGRES_PASSWORD=oasis_dev_password_123

# Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=oasis_redis_dev_123
REDIS_DB=0

# Messaging
KAFKA_BOOTSTRAP_SERVERS=["localhost:9092"]

# Security (generate proper secrets with: make generate-secrets)
SECRET_KEY=dev_secret_key_change_in_production
JWT_SECRET_KEY=dev_jwt_secret_key_change_in_production
EOF
        print_success "Basic .env file created"
    fi
    
    print_warning "Remember to generate secure secrets with: make generate-secrets"
}

setup_database() {
    print_section "Database Setup"
    
    cd "$PROJECT_DIR"
    
    if check_command docker && check_command docker-compose; then
        if docker info >/dev/null 2>&1; then
            print_info "Starting database services with Docker..."
            docker-compose up -d oasis-postgres oasis-redis
            
            # Wait for services to be ready
            print_info "Waiting for database services to be ready..."
            sleep 10
            
            # Run database migrations
            print_info "Running database migrations..."
            poetry run alembic upgrade head || print_warning "Database migrations failed (expected for first run)"
            
            print_success "Database services started"
        else
            print_warning "Docker daemon not running, skipping database setup"
        fi
    else
        print_warning "Docker not available, skipping database setup"
        print_info "Please setup PostgreSQL and Redis manually"
    fi
}

run_tests() {
    print_section "Running Tests"
    
    cd "$PROJECT_DIR"
    
    print_info "Running fast tests..."
    poetry run pytest tests/ -m "not slow" -x --tb=short
    
    if [ $? -eq 0 ]; then
        print_success "Tests passed"
    else
        print_warning "Some tests failed (this is expected for initial setup)"
    fi
}

# =============================================================================
# DEVELOPMENT TOOLS SETUP
# =============================================================================

setup_vscode() {
    print_section "VS Code Configuration"
    
    if [ -d ".vscode" ]; then
        print_info "VS Code configuration already exists"
        return 0
    fi
    
    if prompt_yes_no "Would you like to setup VS Code configuration?"; then
        mkdir -p .vscode
        
        # Create settings.json
        cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.linting.flake8Args": ["--max-line-length=88"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        "htmlcov": true,
        ".coverage": true
    }
}
EOF

        # Create extensions.json
        cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.mypy-type-checker",
        "ms-python.flake8",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "ms-vscode.vscode-docker",
        "ms-kubernetes-tools.vscode-kubernetes-tools"
    ]
}
EOF

        # Create launch.json for debugging
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Trading Engine",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/apps/trading_engine/main.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["-v", "--tb=short"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF

        print_success "VS Code configuration created"
        print_info "Install recommended extensions for better development experience"
    fi
}

create_development_scripts() {
    print_section "Development Scripts"
    
    # Create scripts directory if it doesn't exist
    mkdir -p tools/scripts
    
    # Make the setup script executable
    chmod +x tools/scripts/setup_dev.sh
    
    print_success "Development scripts configured"
}

# =============================================================================
# SUMMARY AND COMPLETION
# =============================================================================

print_summary() {
    print_section "Setup Summary"
    
    echo ""
    echo -e "${GREEN}✅ Development environment setup completed successfully!${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo -e "  1. Generate secure secrets: ${WHITE}make generate-secrets${NC}"
    echo -e "  2. Start infrastructure services: ${WHITE}make docker-up${NC}"
    echo -e "  3. Run database migrations: ${WHITE}make db-upgrade${NC}"
    echo -e "  4. Start the trading engine: ${WHITE}make run-trading-engine${NC}"
    echo -e "  5. Open dashboard: ${WHITE}http://localhost:3000${NC}"
    echo ""
    echo -e "${CYAN}Available commands:${NC}"
    echo -e "  • ${WHITE}make help${NC}           - Show all available commands"
    echo -e "  • ${WHITE}make test${NC}           - Run tests"
    echo -e "  • ${WHITE}make lint${NC}           - Run code quality checks"
    echo -e "  • ${WHITE}make format${NC}         - Format code"
    echo -e "  • ${WHITE}make docker-up${NC}      - Start infrastructure"
    echo -e "  • ${WHITE}make docker-down${NC}    - Stop infrastructure"
    echo ""
    echo -e "${CYAN}Project structure:${NC}"
    echo -e "  • ${WHITE}apps/${NC}              - Core applications"
    echo -e "  • ${WHITE}libs/${NC}              - Shared libraries"  
    echo -e "  • ${WHITE}tools/${NC}             - Development tools"
    echo -e "  • ${WHITE}docs/${NC}              - Documentation"
    echo -e "  • ${WHITE}tests/${NC}             - Test files"
    echo ""
    echo -e "${YELLOW}Important reminders:${NC}"
    echo -e "  ⚠️  Never commit .env files to version control"
    echo -e "  ⚠️  Generate proper secrets before production use"
    echo -e "  ⚠️  Keep API keys and passwords secure"
    echo ""
    echo -e "${MAGENTA}Happy trading! 🚀${NC}"
    echo ""
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    print_header
    
    # System checks
    check_operating_system
    check_python
    check_git
    check_docker
    
    # Installation steps
    install_poetry
    setup_python_environment
    setup_pre_commit
    create_environment_file
    
    # Optional setups
    setup_vscode
    create_development_scripts
    
    # Database and testing
    if prompt_yes_no "Would you like to setup the database now?" "y"; then
        setup_database
    fi
    
    if prompt_yes_no "Would you like to run tests now?" "y"; then
        run_tests
    fi
    
    # Summary
    print_summary
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi