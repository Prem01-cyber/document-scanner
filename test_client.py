# Document Scanner Test Client
# Simple client to test the document scanning API

import requests
import json
import time
import os
from typing import Optional
import argparse

class DocumentScannerClient:
    """Client for testing document scanner API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> bool:
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def quality_check(self, image_path: str) -> dict:
        """Perform quality check only"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = requests.post(
                    f"{self.base_url}/quality-check",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "message": response.text
                }
                
        except requests.RequestException as e:
            return {"error": "Network error", "message": str(e)}
        except FileNotFoundError:
            return {"error": "File not found", "message": f"Could not find {image_path}"}
    
    def scan_document(self, image_path: str) -> dict:
        """Complete document scanning"""
        try:
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = requests.post(
                    f"{self.base_url}/scan-document",
                    files=files,
                    timeout=60
                )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result['client_processing_time'] = end_time - start_time
                return result
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "message": response.text,
                    "client_processing_time": end_time - start_time
                }
                
        except requests.RequestException as e:
            return {"error": "Network error", "message": str(e)}
        except FileNotFoundError:
            return {"error": "File not found", "message": f"Could not find {image_path}"}
    
    def batch_process(self, image_paths: list) -> list:
        """Process multiple documents"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.scan_document(image_path)
            result['image_path'] = image_path
            results.append(result)
            
        return results

def print_quality_result(result: dict):
    """Pretty print quality assessment result"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        print(f"   Message: {result['message']}")
        return
        
    qa = result.get('quality_assessment', {})
    
    print(f"📊 Quality Assessment:")
    print(f"   Needs Rescan: {'❌ YES' if qa.get('needs_rescan') else '✅ NO'}")
    print(f"   Confidence: {qa.get('confidence', 0):.2f}")
    print(f"   Blur Score: {qa.get('blur_score', 0):.1f}")
    print(f"   Brightness: {qa.get('brightness', 0):.1f}")
    
    issues = qa.get('issues', [])
    if issues:
        print(f"   Issues: {', '.join(issues)}")
    else:
        print(f"   Issues: None")

def print_scan_result(result: dict):
    """Pretty print complete scan result"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        print(f"   Message: {result['message']}")
        return
        
    status = result.get('status')
    print(f"📄 Scan Result: {status.upper()}")
    
    if status == "rescan_needed":
        print(f"⚠️  {result.get('message', 'Document needs to be rescanned')}")
        print_quality_result(result)
        return
    
    if status == "success":
        # Quality info
        print_quality_result(result)
        
        # Processing stats
        print(f"\n⏱️  Processing Time:")
        print(f"   Server: {result.get('processing_time_seconds', 0):.2f}s")
        print(f"   Client: {result.get('client_processing_time', 0):.2f}s")
        print(f"   Text Blocks: {result.get('text_blocks_count', 0)}")
        
        # Key-value pairs
        kv_pairs = result.get('key_value_pairs', [])
        print(f"\n🔑 Extracted Key-Value Pairs ({len(kv_pairs)}):")
        
        if not kv_pairs:
            print("   No key-value pairs found")
        else:
            for i, kv in enumerate(kv_pairs, 1):
                print(f"   {i}. {kv['key']} → {kv['value']}")
                print(f"      Confidence: {kv['confidence']:.2f}")
                print(f"      Key Box: ({kv['key_bbox']['x']}, {kv['key_bbox']['y']}) "
                      f"{kv['key_bbox']['width']}×{kv['key_bbox']['height']}")
                print(f"      Value Box: ({kv['value_bbox']['x']}, {kv['value_bbox']['y']}) "
                      f"{kv['value_bbox']['width']}×{kv['value_bbox']['height']}")
                print()

def print_batch_summary(results: list):
    """Print summary of batch processing"""
    total = len(results)
    successful = sum(1 for r in results if r.get('status') == 'success')
    rescan_needed = sum(1 for r in results if r.get('status') == 'rescan_needed')
    errors = sum(1 for r in results if 'error' in r)
    
    print(f"\n📋 Batch Processing Summary:")
    print(f"   Total Documents: {total}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ⚠️  Rescan Needed: {rescan_needed}")
    print(f"   ❌ Errors: {errors}")
    
    if successful > 0:
        avg_kv_pairs = sum(len(r.get('key_value_pairs', [])) 
                          for r in results if r.get('status') == 'success') / successful
        print(f"   📊 Avg Key-Value Pairs: {avg_kv_pairs:.1f}")
        
        processing_times = [r.get('client_processing_time', 0) 
                           for r in results if 'client_processing_time' in r]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"   ⏱️  Avg Processing Time: {avg_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Document Scanner API Test Client")
    parser.add_argument("command", choices=["health", "quality", "scan", "batch"],
                       help="Command to execute")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="API base URL")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--images", nargs="+", help="Paths to multiple image files")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output including all extracted text")
    
    args = parser.parse_args()
    
    client = DocumentScannerClient(args.url)
    
    if args.command == "health":
        print("🏥 Checking API health...")
        if client.health_check():
            print("✅ API is healthy!")
        else:
            print("❌ API is not responding")
            return
    
    elif args.command == "quality":
        if not args.image:
            print("❌ Error: --image required for quality check")
            return
            
        print(f"🔍 Checking quality of: {args.image}")
        result = client.quality_check(args.image)
        print_quality_result(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {args.output}")
    
    elif args.command == "scan":
        if not args.image:
            print("❌ Error: --image required for document scan")
            return
            
        print(f"📄 Scanning document: {args.image}")
        result = client.scan_document(args.image)
        print_scan_result(result)
        
        if args.verbose and result.get('status') == 'success':
            text_blocks = result.get('extracted_text_blocks', [])
            print(f"\n📝 All Extracted Text Blocks ({len(text_blocks)}):")
            for i, block in enumerate(text_blocks, 1):
                print(f"   {i}. \"{block['text'][:50]}{'...' if len(block['text']) > 50 else ''}\"")
                print(f"      Box: ({block['bbox']['x']}, {block['bbox']['y']}) "
                      f"{block['bbox']['width']}×{block['bbox']['height']}")
                print(f"      Confidence: {block['confidence']:.2f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {args.output}")
    
    elif args.command == "batch":
        if not args.images:
            print("❌ Error: --images required for batch processing")
            return
            
        print(f"🔄 Batch processing {len(args.images)} documents...")
        results = client.batch_process(args.images)
        
        # Print individual results
        for i, result in enumerate(results):
            print(f"\n{'='*60}")
            print(f"Document {i+1}: {os.path.basename(result['image_path'])}")
            print('='*60)
            print_scan_result(result)
        
        # Print summary
        print_batch_summary(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Batch results saved to: {args.output}")

# Interactive mode functions
def interactive_mode():
    """Interactive mode for easy testing"""
    client = DocumentScannerClient()
    
    print("🚀 Document Scanner Interactive Test Client")
    print("=" * 50)
    
    # Health check first
    print("Checking API health...")
    if not client.health_check():
        print("❌ API is not responding. Make sure the server is running.")
        return
    print("✅ API is healthy!\n")
    
    while True:
        print("\nCommands:")
        print("1. Quality check")
        print("2. Scan document") 
        print("3. Batch process")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = client.quality_check(image_path)
                print_quality_result(result)
            else:
                print("❌ File not found")
                
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = client.scan_document(image_path)
                print_scan_result(result)
            else:
                print("❌ File not found")
                
        elif choice == "3":
            folder_path = input("Enter folder path: ").strip()
            if os.path.exists(folder_path):
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    import glob
                    image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                    image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
                
                if image_files:
                    print(f"Found {len(image_files)} images")
                    confirm = input("Process all? (y/n): ").strip().lower()
                    if confirm == 'y':
                        results = client.batch_process(image_files)
                        print_batch_summary(results)
                else:
                    print("No image files found")
            else:
                print("❌ Folder not found")
                
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    import sys
    
    # If no arguments provided, run interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()
