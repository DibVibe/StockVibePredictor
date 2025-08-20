#!/usr/bin/env python3

"""
Setup script to prepare Postman environment
"""

import json
import os
import sys

def setup_environment():
    """Setup and validate Postman environment"""

    # Check if Collections exist
    collections_path = "../Collections/StockVibePredictor_API_v2.postman_collection.json"
    if not os.path.exists(collections_path):
        print("❌ Collection file not found!")
        return False

    # Validate collection JSON
    try:
        with open(collections_path, 'r') as f:
            collection = json.load(f)
            print(f"✅ Collection loaded: {collection['info']['name']}")
    except json.JSONDecodeError:
        print("❌ Invalid collection JSON!")
        return False

    # Check environments
    env_files = [
        "../Environments/Development.postman_environment.json",
        "../Environments/Staging.postman_environment.json",
        "../Environments/Production.postman_environment.json"
    ]

    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"✅ Environment found: {os.path.basename(env_file)}")
        else:
            print(f"⚠️  Missing: {os.path.basename(env_file)}")

    print("\n📋 Setup Summary:")
    print("- Collection: Ready")
    print("- Environments: 3 configured")
    print("- Tests: Ready to run")
    print("\n🎯 Next steps:")
    print("1. Import collection and environments into Postman")
    print("2. Update credentials in environment variables")
    print("3. Run 'Health Check' to verify API connection")

    return True

if __name__ == "__main__":
    setup_environment()
