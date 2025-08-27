#!/usr/bin/env python3
"""
Generate secure .env files from .env.example templates

This script:
- Finds all .env.example files in the directory tree
- Replaces password fields with secure random values  
- Uses only shell/database-safe characters
- Preserves all non-password configurations
- Provides dry-run and force options

Created by Claude Code - Docker Scripts Environment Generator
"""

import secrets
import string
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Safe character set for passwords (excludes problematic chars)
SAFE_CHARS = string.ascii_letters + string.digits + "!@#%^*_+-="
PASSWORD_LENGTH = 24

# Color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m' 
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def generate_secure_password() -> str:
    """Generate cryptographically secure password with safe characters only"""
    password = ''.join(secrets.choice(SAFE_CHARS) for _ in range(PASSWORD_LENGTH))
    
    # Verify password contains at least one of each type for good practice
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(c in "!@#%^*_+-=" for c in password)
    
    # Regenerate if missing any type (very rare with 24 chars)
    if not all([has_lower, has_upper, has_digit, has_symbol]):
        return generate_secure_password()
    
    return password

def is_password_field(key: str, value: str) -> bool:
    """Detect if a field should have a password generated"""
    key_upper = key.upper()
    
    # Check for password-related field names
    password_indicators = ['PASSWORD', 'PASSWD', 'PWD']
    has_password_keyword = any(indicator in key_upper for indicator in password_indicators)
    
    # Check for default insecure values
    insecure_values = ['changeme', 'password', 'admin', 'secret', 'default']
    has_insecure_value = value.lower() in insecure_values
    
    return has_password_keyword or has_insecure_value

def parse_env_line(line: str) -> Tuple[str, str, str, str]:
    """Parse a line from .env file
    Returns: (full_line, key, value, comment)
    """
    stripped = line.strip()
    
    # Skip empty lines or comments
    if not stripped or stripped.startswith('#'):
        return line, '', '', ''
    
    # Split on first = sign
    if '=' not in stripped:
        return line, '', '', ''
    
    key_part, value_part = stripped.split('=', 1)
    key = key_part.strip()
    
    # Handle inline comments
    comment = ''
    if '#' in value_part:
        value, comment = value_part.split('#', 1)
        value = value.strip()
        comment = '#' + comment
    else:
        value = value_part.strip()
    
    return line, key, value, comment

def process_env_file(example_path: Path, force: bool = False, dry_run: bool = False) -> Dict:
    """Process a single .env.example file"""
    env_path = example_path.parent / '.env'
    
    # Check if .env already exists
    if env_path.exists() and not force:
        return {
            'status': 'skipped',
            'reason': 'File already exists (use --force to overwrite)',
            'path': env_path
        }
    
    try:
        # Read the example file
        with open(example_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process each line
        new_lines = []
        generated_passwords = []
        preserved_settings = 0
        
        for line in lines:
            original_line, key, value, comment = parse_env_line(line)
            
            if key and value and is_password_field(key, value):
                # Generate new password
                new_password = generate_secure_password()
                new_line = f"{key}={new_password}"
                if comment:
                    new_line += f" {comment}"
                new_line += "\n"
                new_lines.append(new_line)
                generated_passwords.append({
                    'key': key,
                    'old_value': value,
                    'new_value': new_password
                })
            else:
                # Keep original line
                new_lines.append(original_line)
                if key:  # Only count actual settings
                    preserved_settings += 1
        
        # Write the new file (unless dry run)
        if not dry_run:
            with open(env_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        
        return {
            'status': 'success',
            'path': env_path,
            'generated_passwords': generated_passwords,
            'preserved_settings': preserved_settings,
            'dry_run': dry_run
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e),
            'path': env_path
        }

def find_env_example_files(root_path: Path) -> List[Path]:
    """Find all .env.example files recursively"""
    env_files = []
    
    for path in root_path.rglob('.env.example'):
        if path.is_file():
            env_files.append(path)
    
    # Sort for consistent output
    return sorted(env_files)

def print_results(results: List[Dict], dry_run: bool):
    """Print formatted results"""
    
    print(f"{Colors.HEADER}{Colors.BOLD}🔍 Environment File Generation Results{Colors.ENDC}\n")
    
    if dry_run:
        print(f"{Colors.WARNING}📋 DRY RUN MODE - No files were actually created{Colors.ENDC}\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    total_passwords = 0
    
    for result in results:
        status = result['status']
        path = result['path']
        
        if status == 'success':
            success_count += 1
            generated = result['generated_passwords']
            preserved = result['preserved_settings']
            total_passwords += len(generated)
            
            # Print file header
            relative_path = path.relative_to(Path.cwd())
            dry_indicator = " (DRY RUN)" if dry_run else ""
            print(f"{Colors.OKBLUE}📁 {relative_path}{dry_indicator}{Colors.ENDC}")
            
            # Print generated passwords
            for pwd in generated:
                key = pwd['key']
                new_val = pwd['new_value']
                print(f"  {Colors.OKGREEN}✓ Generated {key}: {new_val}{Colors.ENDC}")
            
            # Print preserved settings
            if preserved > 0:
                print(f"  {Colors.OKCYAN}📋 Preserved {preserved} other settings{Colors.ENDC}")
            
            print()  # Blank line
            
        elif status == 'skipped':
            skip_count += 1
            relative_path = path.relative_to(Path.cwd())
            reason = result['reason']
            print(f"{Colors.WARNING}⚠️  Skipped {relative_path}: {reason}{Colors.ENDC}")
            
        elif status == 'error':
            error_count += 1
            relative_path = path.relative_to(Path.cwd())
            reason = result['reason']
            print(f"{Colors.FAIL}❌ Error processing {relative_path}: {reason}{Colors.ENDC}")
    
    # Print summary
    print(f"{Colors.BOLD}📊 Summary:{Colors.ENDC}")
    if success_count > 0:
        print(f"{Colors.OKGREEN}✅ Successfully processed: {success_count} files{Colors.ENDC}")
        print(f"{Colors.OKGREEN}🔐 Generated passwords: {total_passwords}{Colors.ENDC}")
    if skip_count > 0:
        print(f"{Colors.WARNING}⚠️  Skipped: {skip_count} files{Colors.ENDC}")
    if error_count > 0:
        print(f"{Colors.FAIL}❌ Errors: {error_count} files{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate secure .env files from .env.example templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate-env.py                     # Generate .env files
  python3 generate-env.py --dry-run          # Preview what would be generated  
  python3 generate-env.py --force            # Overwrite existing .env files
  python3 generate-env.py --path ./mysql     # Only process specific directory
        """
    )
    
    parser.add_argument('--force', '-f', action='store_true',
                       help='Overwrite existing .env files')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Preview changes without creating files')
    parser.add_argument('--path', '-p', type=str, default='.',
                       help='Root path to search for .env.example files (default: current directory)')
    
    args = parser.parse_args()
    
    # Convert path to Path object and validate
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"{Colors.FAIL}❌ Error: Path '{args.path}' does not exist{Colors.ENDC}")
        return 1
    
    if not root_path.is_dir():
        print(f"{Colors.FAIL}❌ Error: Path '{args.path}' is not a directory{Colors.ENDC}")
        return 1
    
    # Find all .env.example files
    env_example_files = find_env_example_files(root_path)
    
    if not env_example_files:
        print(f"{Colors.WARNING}⚠️  No .env.example files found in '{root_path}'{Colors.ENDC}")
        return 0
    
    print(f"{Colors.HEADER}🔍 Found {len(env_example_files)} .env.example files{Colors.ENDC}\n")
    
    # Process each file
    results = []
    for env_file in env_example_files:
        result = process_env_file(env_file, force=args.force, dry_run=args.dry_run)
        results.append(result)
    
    # Print results
    print_results(results, args.dry_run)
    
    # Return appropriate exit code
    error_count = sum(1 for r in results if r['status'] == 'error')
    return 1 if error_count > 0 else 0

if __name__ == '__main__':
    exit(main())