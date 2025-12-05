#!/usr/bin/env python3
"""
Comprehensive test runner for LEAF-YOLO
Provides different testing modes and reporting options
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and capture output."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        
        print(f"✅ {description} completed successfully in {duration:.2f}s")
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        
        print(f"❌ {description} failed after {duration:.2f}s")
        print(f"Exit code: {e.returncode}")
        
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
            
        if e.stderr:
            print("\nStderr:")
            print(e.stderr)
            
        return False


def run_unit_tests(verbose=True, coverage=False):
    """Run unit tests."""
    cmd = ['pytest', 'tests/unit/', '-v' if verbose else '-q']
    
    if coverage:
        cmd.extend(['--cov=leafyolo', '--cov-report=term-missing'])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=True):
    """Run integration tests."""
    cmd = ['pytest', 'tests/integration/', '-v' if verbose else '-q']
    return run_command(cmd, "Integration Tests")


def run_benchmark_tests():
    """Run benchmark tests."""
    cmd = ['pytest', 'tests/benchmarks/', '-v', '--benchmark-only']
    return run_command(cmd, "Performance Benchmarks")


def run_fast_tests():
    """Run fast tests only (exclude slow and benchmarks)."""
    cmd = ['pytest', 'tests/', '-v', '-m', 'not slow and not benchmark']
    return run_command(cmd, "Fast Tests")


def run_all_tests(coverage=False):
    """Run all tests."""
    cmd = ['pytest', 'tests/', '-v']
    
    if coverage:
        cmd.extend(['--cov=leafyolo', '--cov-report=html', '--cov-report=term'])
    
    return run_command(cmd, "All Tests")


def run_linting():
    """Run code linting."""
    results = []
    
    # Flake8
    cmd = ['flake8', 'leafyolo', 'tests', '--max-line-length=100', 
           '--ignore=E203,W503,E501']
    results.append(run_command(cmd, "Flake8 Linting"))
    
    # MyPy (if available)
    try:
        cmd = ['mypy', 'leafyolo', '--ignore-missing-imports']
        results.append(run_command(cmd, "MyPy Type Checking"))
    except FileNotFoundError:
        print("⚠️  MyPy not available, skipping type checking")
    
    return all(results)


def run_formatting_check():
    """Check code formatting."""
    results = []
    
    # Black
    cmd = ['black', '--check', 'leafyolo', 'tests']
    results.append(run_command(cmd, "Black Format Check"))
    
    # isort
    cmd = ['isort', '--check', 'leafyolo', 'tests']
    results.append(run_command(cmd, "Import Sort Check"))
    
    return all(results)


def check_dependencies():
    """Check if all test dependencies are available."""
    required_packages = [
        'pytest', 'pytest-cov', 'pytest-benchmark'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def generate_test_report():
    """Generate comprehensive test report."""
    report_dir = Path('test_reports')
    report_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"test_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"LEAF-YOLO Test Report - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Run tests with output capture
        cmd = ['pytest', 'tests/', '-v', '--tb=short']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        f.write("Test Results:\n")
        f.write("-" * 20 + "\n")
        f.write(result.stdout)
        
        if result.stderr:
            f.write("\n\nErrors:\n")
            f.write("-" * 20 + "\n")
            f.write(result.stderr)
    
    print(f"📄 Test report saved to: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(
        description="LEAF-YOLO Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py --fast                    # Run fast tests only
  python tests/run_tests.py --unit --coverage         # Unit tests with coverage
  python tests/run_tests.py --integration             # Integration tests
  python tests/run_tests.py --benchmark               # Performance benchmarks
  python tests/run_tests.py --all --coverage          # All tests with coverage
  python tests/run_tests.py --lint --format           # Code quality checks
  python tests/run_tests.py --ci                      # CI pipeline (fast + lint)
        """
    )
    
    # Test selection
    parser.add_argument('--unit', action='store_true', 
                       help='Run unit tests')
    parser.add_argument('--integration', action='store_true', 
                       help='Run integration tests')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark tests')
    parser.add_argument('--fast', action='store_true', 
                       help='Run fast tests only (no slow/benchmark)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests')
    
    # Code quality
    parser.add_argument('--lint', action='store_true', 
                       help='Run linting checks')
    parser.add_argument('--format', action='store_true', 
                       help='Check code formatting')
    
    # Options
    parser.add_argument('--coverage', action='store_true', 
                       help='Generate coverage report')
    parser.add_argument('--quiet', action='store_true', 
                       help='Run tests in quiet mode')
    parser.add_argument('--report', action='store_true', 
                       help='Generate test report')
    
    # Presets
    parser.add_argument('--ci', action='store_true', 
                       help='Run CI pipeline (fast tests + linting)')
    parser.add_argument('--full', action='store_true', 
                       help='Run full test suite (all + lint + format + coverage)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    print("🧪 LEAF-YOLO Test Runner")
    print(f"Working directory: {Path.cwd()}")
    
    results = []
    start_total = time.time()
    
    # Handle presets
    if args.ci:
        print("\n🔄 Running CI Pipeline...")
        results.append(run_fast_tests())
        results.append(run_linting())
    
    elif args.full:
        print("\n🔄 Running Full Test Suite...")
        results.append(run_all_tests(coverage=args.coverage or True))
        results.append(run_linting())
        results.append(run_formatting_check())
    
    else:
        # Individual test categories
        if args.unit:
            results.append(run_unit_tests(
                verbose=not args.quiet, 
                coverage=args.coverage
            ))
        
        if args.integration:
            results.append(run_integration_tests(verbose=not args.quiet))
        
        if args.benchmark:
            results.append(run_benchmark_tests())
        
        if args.fast:
            results.append(run_fast_tests())
        
        if args.all:
            results.append(run_all_tests(coverage=args.coverage))
        
        # Code quality checks
        if args.lint:
            results.append(run_linting())
        
        if args.format:
            results.append(run_formatting_check())
    
    # Generate report if requested
    if args.report:
        generate_test_report()
    
    # Summary
    total_duration = time.time() - start_total
    
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Tests run: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
