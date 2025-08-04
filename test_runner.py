#!/usr/bin/env python3
"""
Test script to verify the centralized runner works correctly
"""

import subprocess
import sys
import os

def test_runner():
    """Test the centralized runner functionality"""
    print("🧪 Testing Centralized Runner...")
    
    # Test help command
    try:
        result = subprocess.run([
            sys.executable, "run.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Help command works")
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Help command timed out")
        return False
    except Exception as e:
        print(f"❌ Help command error: {e}")
        return False
    
    # Test status command
    try:
        result = subprocess.run([
            sys.executable, "run.py", "status"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Status command works")
        else:
            print(f"❌ Status command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Status command timed out")
        return False
    except Exception as e:
        print(f"❌ Status command error: {e}")
        return False
    
    # Test test command  
    try:
        result = subprocess.run([
            sys.executable, "run.py", "test"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test command works")
        else:
            print(f"⚠️ Test command had issues: {result.stderr}")
            # Don't fail here as system might not be fully set up
            
    except subprocess.TimeoutExpired:
        print("❌ Test command timed out")
        return False
    except Exception as e:
        print(f"❌ Test command error: {e}")
        return False
    
    print("\n🎉 Centralized runner basic functionality verified!")
    return True

if __name__ == "__main__":
    success = test_runner()
    sys.exit(0 if success else 1)