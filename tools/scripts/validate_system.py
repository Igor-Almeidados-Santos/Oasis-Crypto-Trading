#!/usr/bin/env python3
"""
Oasis Crypto Trade - System Validation Script
==============================================

Comprehensive validation of the Oasis trading system setup.

Features:
- Infrastructure service health checks
- Configuration validation
- Dependency verification
- Performance baseline testing
- Security configuration checks
- File structure validation

Author: Oasis Trading Systems
License: Proprietary
"""

import asyncio
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Color constants for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ValidationResult:
    """Result of a validation check."""
    
    def __init__(
        self,
        name: str,
        passed: bool,
        message: str,
        details: Optional[Dict] = None,
        execution_time: Optional[float] = None,
        critical: bool = False
    ):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.execution_time = execution_time
        self.critical = critical
        self.timestamp = datetime.utcnow()


class SystemValidator:
    """Main system validation class."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.project_root = Path(__file__).parent.parent.parent
        self.critical_failures = 0
        self.total_checks = 0
        
    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
        self.total_checks += 1
        
        if result.critical and not result.passed:
            self.critical_failures += 1
    
    def print_header(self):
        """Print validation header."""
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}🏛️  OASIS CRYPTO TRADE - SYSTEM VALIDATION{Colors.END}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.END}\n")
    
    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{Colors.BLUE}▶ {title}{Colors.END}")
        print(f"{Colors.BLUE}{'-' * 50}{Colors.END}")
    
    def print_result(self, result: ValidationResult):
        """Print individual validation result."""
        status_icon = "✅" if result.passed else "❌"
        status_color = Colors.GREEN if result.passed else Colors.RED
        critical_mark = " [CRITICAL]" if result.critical else ""
        time_info = f" ({result.execution_time:.3f}s)" if result.execution_time else ""
        
        print(f"{status_color}{status_icon} {result.name}{critical_mark}{time_info}{Colors.END}")
        if result.message:
            print(f"   {result.message}")
        
        if result.details and not result.passed:
            for key, value in result.details.items():
                print(f"   {Colors.YELLOW}• {key}: {value}{Colors.END}")
    
    # ==========================================================================
    # INFRASTRUCTURE VALIDATIONS
    # ==========================================================================
    
    def validate_project_structure(self) -> ValidationResult:
        """Validate project directory structure."""
        start_time = time.time()
        
        required_dirs = [
            "apps/trading_engine",
            "apps/market_data_service", 
            "apps/risk_management",
            "apps/analytics_service",
            "apps/web_dashboard",
            "libs/shared",
            "libs/domain",
            "libs/infrastructure",
            "libs/strategies",
            "tools/scripts",
            "tools/testing",
            "docs",
            "infra",
            "docker",
            "kubernetes"
        ]
        
        required_files = [
            "pyproject.toml",
            "docker-compose.yml",
            "Makefile",
            "README.md",
            ".env.example",
            ".pre-commit-config.yaml",
            "pytest.ini"
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        execution_time = time.time() - start_time
        
        if missing_dirs or missing_files:
            details = {}
            if missing_dirs:
                details["missing_directories"] = missing_dirs
            if missing_files:
                details["missing_files"] = missing_files
            
            return ValidationResult(
                "Project Structure",
                False,
                f"Missing {len(missing_dirs)} directories and {len(missing_files)} files",
                details,
                execution_time,
                critical=True
            )
        
        return ValidationResult(
            "Project Structure",
            True,
            f"All {len(required_dirs)} directories and {len(required_files)} files present",
            {"total_dirs": len(required_dirs), "total_files": len(required_files)},
            execution_time
        )
    
    def validate_python_dependencies(self) -> ValidationResult:
        """Validate Python dependencies."""
        start_time = time.time()
        
        try:
            import poetry.core.pyproject.toml as poetry_toml
            
            pyproject_path = self.project_root / "pyproject.toml"
            if not pyproject_path.exists():
                return ValidationResult(
                    "Python Dependencies",
                    False,
                    "pyproject.toml not found",
                    execution_time=time.time() - start_time,
                    critical=True
                )
            
            # Check critical dependencies
            critical_deps = [
                "fastapi", "uvicorn", "sqlalchemy", "redis", "aiokafka",
                "pydantic", "asyncpg", "structlog", "pytest"
            ]
            
            missing_deps = []
            
            try:
                for dep in critical_deps:
                    try:
                        __import__(dep.replace("-", "_"))
                    except ImportError:
                        missing_deps.append(dep)
                
                execution_time = time.time() - start_time
                
                if missing_deps:
                    return ValidationResult(
                        "Python Dependencies",
                        False,
                        f"Missing {len(missing_deps)} critical dependencies",
                        {"missing_dependencies": missing_deps},
                        execution_time,
                        critical=True
                    )
                
                return ValidationResult(
                    "Python Dependencies",
                    True,
                    f"All {len(critical_deps)} critical dependencies available",
                    {"checked_dependencies": len(critical_deps)},
                    execution_time
                )
                
            except Exception as e:
                return ValidationResult(
                    "Python Dependencies",
                    False,
                    f"Error checking dependencies: {str(e)}",
                    execution_time=time.time() - start_time
                )
                
        except ImportError:
            return ValidationResult(
                "Python Dependencies",
                False,
                "Poetry not available for dependency checking",
                execution_time=time.time() - start_time
            )
    
    async def validate_database_connection(self) -> ValidationResult:
        """Validate database connection."""
        start_time = time.time()
        
        try:
            from libs.infrastructure.database.connection import test_database_connection
            from libs.shared.config.base import get_settings
            
            settings = get_settings()
            success = await test_database_connection(settings)
            
            execution_time = time.time() - start_time
            
            if success:
                return ValidationResult(
                    "Database Connection",
                    True,
                    "Database connection successful",
                    {"host": settings.POSTGRES_HOST, "port": settings.POSTGRES_PORT},
                    execution_time
                )
            else:
                return ValidationResult(
                    "Database Connection",
                    False,
                    "Database connection failed",
                    {"host": settings.POSTGRES_HOST, "port": settings.POSTGRES_PORT},
                    execution_time,
                    critical=True
                )
        
        except Exception as e:
            return ValidationResult(
                "Database Connection",
                False,
                f"Database validation error: {str(e)}",
                {"error": str(e)},
                time.time() - start_time,
                critical=True
            )
    
    async def validate_cache_connection(self) -> ValidationResult:
        """Validate cache connection."""
        start_time = time.time()
        
        try:
            from libs.infrastructure.cache.redis_client import test_cache_connection
            from libs.shared.config.base import get_settings
            
            settings = get_settings()
            success = await test_cache_connection(settings)
            
            execution_time = time.time() - start_time
            
            if success:
                return ValidationResult(
                    "Cache Connection",
                    True,
                    "Cache connection successful",
                    {"host": settings.REDIS_HOST, "port": settings.REDIS_PORT},
                    execution_time
                )
            else:
                return ValidationResult(
                    "Cache Connection",
                    False,
                    "Cache connection failed",
                    {"host": settings.REDIS_HOST, "port": settings.REDIS_PORT},
                    execution_time
                )
        
        except Exception as e:
            return ValidationResult(
                "Cache Connection",
                False,
                f"Cache validation error: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
    
    async def validate_messaging_system(self) -> ValidationResult:
        """Validate messaging system."""
        start_time = time.time()
        
        try:
            from libs.infrastructure.messaging.kafka_producer import test_kafka_producer
            from libs.shared.config.base import get_settings
            
            settings = get_settings()
            success = await test_kafka_producer(settings)
            
            execution_time = time.time() - start_time
            
            if success:
                return ValidationResult(
                    "Messaging System",
                    True,
                    "Kafka producer test successful",
                    {"bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS},
                    execution_time
                )
            else:
                return ValidationResult(
                    "Messaging System",
                    False,
                    "Kafka producer test failed",
                    {"bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS},
                    execution_time
                )
        
        except Exception as e:
            return ValidationResult(
                "Messaging System",
                False,
                f"Messaging validation error: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
    
    # ==========================================================================
    # CONFIGURATION VALIDATIONS
    # ==========================================================================
    
    def validate_environment_configuration(self) -> ValidationResult:
        """Validate environment configuration."""
        start_time = time.time()
        
        try:
            from libs.shared.config.base import get_settings, validate_configuration
            
            settings = get_settings()
            issues = validate_configuration(settings)
            
            execution_time = time.time() - start_time
            
            if issues:
                return ValidationResult(
                    "Environment Configuration",
                    False,
                    f"Found {len(issues)} configuration issues",
                    {"issues": issues},
                    execution_time,
                    critical=True
                )
            
            return ValidationResult(
                "Environment Configuration",
                True,
                "Configuration validation passed",
                {"environment": settings.ENVIRONMENT},
                execution_time
            )
        
        except Exception as e:
            return ValidationResult(
                "Environment Configuration",
                False,
                f"Configuration validation error: {str(e)}",
                {"error": str(e)},
                time.time() - start_time,
                critical=True
            )
    
    def validate_security_configuration(self) -> ValidationResult:
        """Validate security configuration."""
        start_time = time.time()
        
        try:
            from libs.shared.config.base import get_settings
            
            settings = get_settings()
            issues = []
            
            # Check secret key strength
            if len(settings.SECRET_KEY.get_secret_value()) < 32:
                issues.append("SECRET_KEY is too short (minimum 32 characters)")
            
            # Check JWT secret
            if len(settings.JWT_SECRET_KEY.get_secret_value()) < 32:
                issues.append("JWT_SECRET_KEY is too short (minimum 32 characters)")
            
            # Check production settings
            if settings.is_production:
                if settings.DEBUG:
                    issues.append("DEBUG should be False in production")
                
                if "example" in settings.SECRET_KEY.get_secret_value().lower():
                    issues.append("Using example SECRET_KEY in production")
            
            execution_time = time.time() - start_time
            
            if issues:
                return ValidationResult(
                    "Security Configuration",
                    False,
                    f"Found {len(issues)} security issues",
                    {"issues": issues},
                    execution_time,
                    critical=settings.is_production
                )
            
            return ValidationResult(
                "Security Configuration",
                True,
                "Security configuration validated",
                {"environment": settings.ENVIRONMENT},
                execution_time
            )
        
        except Exception as e:
            return ValidationResult(
                "Security Configuration",
                False,
                f"Security validation error: {str(e)}",
                {"error": str(e)},
                time.time() - start_time,
                critical=True
            )
    
    # ==========================================================================
    # PERFORMANCE VALIDATIONS
    # ==========================================================================
    
    def validate_system_performance(self) -> ValidationResult:
        """Validate system performance baseline."""
        start_time = time.time()
        
        try:
            import psutil
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_available_gb = memory.available / (1024**3)
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            issues = []
            warnings = []
            
            # Performance thresholds
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 60:
                warnings.append(f"Moderate CPU usage: {cpu_percent:.1f}%")
            
            if memory_available_gb < 1.0:
                issues.append(f"Low memory: {memory_available_gb:.1f}GB available")
            elif memory_available_gb < 2.0:
                warnings.append(f"Limited memory: {memory_available_gb:.1f}GB available")
            
            if disk_free_gb < 5.0:
                issues.append(f"Low disk space: {disk_free_gb:.1f}GB free")
            elif disk_free_gb < 10.0:
                warnings.append(f"Limited disk space: {disk_free_gb:.1f}GB free")
            
            execution_time = time.time() - start_time
            
            details = {
                "cpu_usage_percent": cpu_percent,
                "memory_available_gb": round(memory_available_gb, 2),
                "disk_free_gb": round(disk_free_gb, 2)
            }
            
            if issues:
                details["issues"] = issues
                return ValidationResult(
                    "System Performance",
                    False,
                    f"Performance issues detected: {', '.join(issues)}",
                    details,
                    execution_time
                )
            
            message = "System performance acceptable"
            if warnings:
                message += f" (warnings: {', '.join(warnings)})"
                details["warnings"] = warnings
            
            return ValidationResult(
                "System Performance",
                True,
                message,
                details,
                execution_time
            )
        
        except ImportError:
            return ValidationResult(
                "System Performance",
                False,
                "psutil not available for performance monitoring",
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return ValidationResult(
                "System Performance",
                False,
                f"Performance validation error: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
    
    # ==========================================================================
    # MAIN VALIDATION METHODS
    # ==========================================================================
    
    async def run_all_validations(self) -> Tuple[int, int]:
        """Run all system validations."""
        self.print_header()
        
        # Infrastructure validations
        self.print_section("Infrastructure Validation")
        
        self.add_result(self.validate_project_structure())
        self.print_result(self.results[-1])
        
        self.add_result(self.validate_python_dependencies())
        self.print_result(self.results[-1])
        
        # Async validations
        db_result = await self.validate_database_connection()
        self.add_result(db_result)
        self.print_result(self.results[-1])
        
        cache_result = await self.validate_cache_connection()
        self.add_result(cache_result)
        self.print_result(self.results[-1])
        
        messaging_result = await self.validate_messaging_system()
        self.add_result(messaging_result)
        self.print_result(self.results[-1])
        
        # Configuration validations
        self.print_section("Configuration Validation")
        
        self.add_result(self.validate_environment_configuration())
        self.print_result(self.results[-1])
        
        self.add_result(self.validate_security_configuration())
        self.print_result(self.results[-1])
        
        # Performance validations
        self.print_section("Performance Validation")
        
        self.add_result(self.validate_system_performance())
        self.print_result(self.results[-1])
        
        return self.print_summary()
    
    def print_summary(self) -> Tuple[int, int]:
        """Print validation summary."""
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = self.total_checks - passed_checks
        
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}VALIDATION SUMMARY{Colors.END}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.END}")
        
        # Overall status
        if failed_checks == 0:
            status_color = Colors.GREEN
            status_icon = "✅"
            status_text = "ALL CHECKS PASSED"
        elif self.critical_failures == 0:
            status_color = Colors.YELLOW
            status_icon = "⚠️"
            status_text = "PASSED WITH WARNINGS"
        else:
            status_color = Colors.RED
            status_icon = "❌"
            status_text = "CRITICAL FAILURES DETECTED"
        
        print(f"\n{status_color}{status_icon} {status_text}{Colors.END}")
        
        # Statistics
        print(f"\nResults:")
        print(f"  {Colors.GREEN}✅ Passed: {passed_checks}{Colors.END}")
        print(f"  {Colors.RED}❌ Failed: {failed_checks}{Colors.END}")
        print(f"  {Colors.YELLOW}🔥 Critical: {self.critical_failures}{Colors.END}")
        print(f"  📊 Total: {self.total_checks}")
        
        # Execution time
        total_time = sum(r.execution_time for r in self.results if r.execution_time)
        print(f"  ⏱️ Total time: {total_time:.3f}s")
        
        # Critical failures details
        if self.critical_failures > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}CRITICAL FAILURES:{Colors.END}")
            for result in self.results:
                if result.critical and not result.passed:
                    print(f"  {Colors.RED}❌ {result.name}: {result.message}{Colors.END}")
        
        # Next steps
        if failed_checks == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 System is ready for development!{Colors.END}")
            print(f"\nNext steps:")
            print(f"  1. Start infrastructure: {Colors.WHITE}make docker-up{Colors.END}")
            print(f"  2. Run database migrations: {Colors.WHITE}make db-upgrade{Colors.END}")
            print(f"  3. Start trading engine: {Colors.WHITE}make run-trading-engine{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}🔧 Please fix the issues above before proceeding{Colors.END}")
            
            if self.critical_failures > 0:
                print(f"{Colors.RED}⚠️  Critical failures must be resolved{Colors.END}")
        
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}\n")
        
        return passed_checks, failed_checks


async def main():
    """Main validation function."""
    validator = SystemValidator()
    
    try:
        # Setup logging for the script
        import logging
        logging.basicConfig(level=logging.WARNING)
        
        passed, failed = await validator.run_all_validations()
        
        # Exit codes
        if validator.critical_failures > 0:
            sys.exit(2)  # Critical failures
        elif failed > 0:
            sys.exit(1)  # Non-critical failures
        else:
            sys.exit(0)  # All good
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {str(e)}{Colors.END}")
        if validator.results:
            print(f"\nCompleted {len(validator.results)} checks before error")
        
        # Print traceback in debug mode
        if os.getenv('DEBUG', '').lower() == 'true':
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())