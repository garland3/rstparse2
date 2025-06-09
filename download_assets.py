#!/usr/bin/env python3
"""
Download static assets for the RAG app to avoid CDN dependencies.
This script downloads JavaScript and CSS files needed for the chat interface.
"""

import requests
import os
from pathlib import Path
import sys

def download_file(url, local_path):
    """Download a file from URL to local path"""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        file_size = len(response.content)
        print(f"✓ Downloaded: {local_path} ({file_size} bytes)")
        return True
        
    except requests.RequestException as e:
        print(f"✗ Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error saving {local_path}: {e}")
        return False

def main():
    """Download all required static assets"""
    # Asset URLs and their local paths
    assets = {
        'https://cdn.jsdelivr.net/npm/marked/marked.min.js': 'static/js/marked.min.js',
        'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js': 'static/js/prism-core.min.js',
        'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js': 'static/js/prism-autoloader.min.js',
        'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.css': 'static/css/prism.css'
    }
    
    print("Starting download of static assets...")
    print("=" * 50)
    
    success_count = 0
    total_count = len(assets)
    
    for url, path in assets.items():
        if download_file(url, path):
            success_count += 1
    
    print("=" * 50)
    print(f"Download complete: {success_count}/{total_count} files downloaded successfully")
    
    if success_count == total_count:
        print("✓ All assets downloaded successfully!")
        return 0
    else:
        print("✗ Some downloads failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
