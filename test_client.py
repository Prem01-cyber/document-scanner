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
        if "message" in result:
            print(f"   Message: {result['message']}")
        
        # Try to extract more detailed error info if available
        if isinstance(result.get('message'), str):
            try:
                import json
                error_detail = json.loads(result['message'])
                if 'detail' in error_detail:
                    print(f"   Details: {error_detail['detail']}")
            except (json.JSONDecodeError, TypeError):
                pass
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
        if "message" in result:
            print(f"   Message: {result['message']}")
        
        # Try to extract more detailed error info if available  
        if isinstance(result.get('message'), str):
            try:
                import json
                error_detail = json.loads(result['message'])
                if 'detail' in error_detail:
                    print(f"   Details: {error_detail['detail']}")
                    # Check for specific error types
                    if "JSON serializable" in error_detail['detail']:
                        print(f"   🔧 This appears to be a server-side data type conversion issue.")
                        print(f"   💡 Try restarting the server or check server logs for more details.")
            except (json.JSONDecodeError, TypeError):
                pass
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
        
        # Show enhanced splitting statistics
        extraction_stats = result.get('extraction_statistics', {})
        if extraction_stats:
            print(f"\n🔄 Hybrid Text Analysis Results:")
            print(f"   Original OCR Blocks: {extraction_stats.get('original_ocr_blocks', 0)}")
            
            pairs_from_splitting = extraction_stats.get('pairs_from_splitting', 0)
            if pairs_from_splitting > 0:
                print(f"   ✅ Pairs from Text Splitting: {pairs_from_splitting}")
                print(f"   🎯 Splitting Success Rate: {(pairs_from_splitting / max(1, extraction_stats.get('original_ocr_blocks', 1))) * 100:.1f}%")
            else:
                print(f"   ❌ No pairs from text splitting")
                
            print(f"   📊 Total Pairs Extracted: {extraction_stats.get('identified_key_value_pairs', 0)}")
            
            # Show extraction methods used
            methods_used = extraction_stats.get('extraction_methods_used', [])
            if methods_used:
                print(f"   🔧 Extraction Methods: {', '.join(methods_used[:3])}{'...' if len(methods_used) > 3 else ''}")
            
            avg_confidence = extraction_stats.get('average_confidence', 0)
            if avg_confidence > 0:
                print(f"   📈 Average Confidence: {avg_confidence:.2f}")
                
        # Debugging hint for server logs
        if result.get('text_blocks_count', 0) > 0 and extraction_stats.get('identified_key_value_pairs', 0) == 0:
            print(f"\n🔧 Debugging Tips:")
            print(f"   • Check server console/logs for detailed splitting output")
            print(f"   • Look for '=== STARTING TEXT BLOCK SPLITTING ===' messages")
            print(f"   • Text blocks detected but no pairs extracted - check splitting logic")
        
        # Enhanced display of extracted text blocks with splitting indicators
        text_blocks = result.get('extracted_text_blocks', [])
        print(f"\n📝 All Extracted Text Blocks ({len(text_blocks)}):")
        
        # Categorize blocks by confidence to show which might be from splitting
        original_blocks = []
        split_blocks = []
        
        for i, block in enumerate(text_blocks, 1):
            # Blocks with confidence 0.95 are likely from keyword splitting
            # Blocks with confidence 0.90 are likely split values
            is_likely_split = block['confidence'] in [0.95, 0.90]
            
            # Truncate long text for readability
            display_text = block['text'][:100] + '...' if len(block['text']) > 100 else block['text']
            
            # Add indicators for split blocks
            split_indicator = " 🔄" if is_likely_split else ""
            
            print(f"   {i}. \"{display_text}\"{split_indicator}")
            print(f"      Box: ({block['bbox']['x']}, {block['bbox']['y']}) "
                  f"{block['bbox']['width']}×{block['bbox']['height']}")
            print(f"      Confidence: {block['confidence']:.2f}")
            
            if is_likely_split:
                split_blocks.append(block)
            else:
                original_blocks.append(block)
            print()
        
        # Show splitting analysis
        if len(split_blocks) > 0:
            print(f"   📊 Block Analysis:")
            print(f"      Original OCR Blocks: {len(original_blocks)}")
            print(f"      Likely Split Blocks: {len(split_blocks)} 🔄")
            print(f"      Total Blocks: {len(text_blocks)}")
        
        # Analyze block patterns dynamically
        form_field_patterns = []
        value_patterns = []
        
        for block in text_blocks:
            text = block['text']
            # Look for title case patterns that might be form fields
            if text.istitle() and len(text.split()) <= 3 and block['confidence'] >= 0.95:
                form_field_patterns.append(text)
            elif text.istitle() and len(text.split()) <= 2 and block['confidence'] >= 0.90:
                value_patterns.append(text)
        
        if form_field_patterns or value_patterns:
            print(f"\n🎯 Pattern Analysis:")
            if form_field_patterns:
                print(f"      📋 Potential Form Fields: {', '.join(form_field_patterns[:3])}{'...' if len(form_field_patterns) > 3 else ''}")
            if value_patterns:
                print(f"      📝 Potential Values: {', '.join(value_patterns[:3])}{'...' if len(value_patterns) > 3 else ''}")
        else:
            print(f"\n❌ No recognizable form patterns found")
        
        # Enhanced key-value pairs display
        kv_pairs = result.get('key_value_pairs', [])
        print(f"\n🔑 Extracted Key-Value Pairs ({len(kv_pairs)}):")
        
        if not kv_pairs:
            print("   ❌ No key-value pairs found")
            
            # If we have text blocks but no pairs, provide more analysis
            if len(text_blocks) > 0:
                print(f"\n💡 Analysis:")
                print(f"   • {len(text_blocks)} text blocks were detected")
                print(f"   • Check if text splitting is working correctly")
                print(f"   • Look for server console output showing splitting process")
                
                # Check if we have potential combined blocks (form field + value patterns)
                combined_blocks = [block for block in text_blocks if 
                                 len(block['text'].split()) >= 2 and 
                                 block['text'].istitle() and
                                 block['confidence'] >= 0.95]
                if combined_blocks:
                    print(f"   • Found {len(combined_blocks)} blocks with potential form field patterns")
                    print(f"   • These should be getting split automatically")
        else:
            for i, kv in enumerate(kv_pairs, 1):
                # Add method indicator
                method = kv.get('extraction_method', '')
                method_indicator = ""
                if 'split' in method.lower():
                    method_indicator = " 🔄"
                elif 'manual' in method.lower():
                    method_indicator = " 🎯"
                elif 'spatial' in method.lower():
                    method_indicator = " 📍"
                
                print(f"   {i}. {kv['key']} → {kv['value']}{method_indicator}")
                print(f"      Confidence: {kv['confidence']:.2f}")
                
                # Show extraction method if available
                if method:
                    # Simplify method display
                    simplified_method = method.replace('|', ' | ').replace('_', ' ').title()
                    print(f"      Method: {simplified_method}")
                
                print(f"      Key Box: ({kv['key_bbox']['x']}, {kv['key_bbox']['y']}) "
                      f"{kv['key_bbox']['width']}×{kv['key_bbox']['height']}")
                print(f"      Value Box: ({kv['value_bbox']['x']}, {kv['value_bbox']['y']}) "
                      f"{kv['value_bbox']['width']}×{kv['value_bbox']['height']}")
                print()
            
            # Summary of extraction methods
            if len(kv_pairs) > 1:
                methods = [kv.get('extraction_method', '') for kv in kv_pairs]
                split_methods = sum(1 for m in methods if 'split' in m.lower())
                if split_methods > 0:
                    print(f"   📊 Summary: {split_methods}/{len(kv_pairs)} pairs from text splitting")

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
    parser = argparse.ArgumentParser(description="Document Scanner API Test Client with Enhanced Debugging")
    parser.add_argument("command", choices=["health", "quality", "scan", "batch", "debug"],
                       help="Command to execute")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="API base URL")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--images", nargs="+", help="Paths to multiple image files")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output including all extracted text")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Show debug information and troubleshooting tips")
    
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
    
    elif args.command == "debug":
        print("\n🔧 Document Scanner Debug Information")
        print("="*60)
        print("Enhanced debugging features have been added to help troubleshoot")
        print("text splitting and key-value extraction issues.")
        
        print("\n📊 What to Look For:")
        print("• Server console output showing text block splitting process")
        print("• Manual split detection for common form patterns") 
        print("• Confidence scores indicating split vs original blocks")
        print("• Extraction method details in key-value pairs")
        
        print("\n🎯 Dynamic Processing:")
        print("Text blocks containing form fields + values get intelligently split using:")
        print("• NLP analysis for semantic understanding")
        print("• POS tagging to identify label vs value patterns")  
        print("• Spatial analysis for accurate key-value pairing")
        
        print("\n💡 Troubleshooting Steps:")
        print("1. Restart the server to load new debug code")
        print("2. Check server console for splitting messages")
        print("3. Look for confidence values 0.95 and 0.90 (split blocks)")
        print("4. Use interactive mode (no args) for best debugging experience")
        
        if args.debug:
            print("\n🔍 Extended Debug Mode Active")
            print("Run 'scan' command with --debug for additional output")
            
        print(f"\n🌐 API URL: {args.url}")
        print("💡 Use interactive mode for step-by-step debugging")

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
        print("4. Debug mode info")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = client.quality_check(image_path)
                print_quality_result(result)
                
                # If there's a server error, provide troubleshooting tips
                if "error" in result and "HTTP 500" in str(result.get('error', '')):
                    print("\n🛠️  Troubleshooting Tips:")
                    print("   • The server may need to be restarted")
                    print("   • Check server logs for detailed error information") 
                    print("   • Ensure all dependencies are properly installed")
                    print("   • Try a different image format if the issue persists")
            else:
                print("❌ File not found")
                
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = client.scan_document(image_path)
                print_scan_result(result)
                
                # If there's a server error, provide troubleshooting tips
                if "error" in result and "HTTP 500" in str(result.get('error', '')):
                    print("\n🛠️  Troubleshooting Tips:")
                    print("   • The server may need to be restarted")
                    print("   • Check server logs for detailed error information")
                    print("   • Ensure all dependencies are properly installed")
                    print("   • Try a different image format if the issue persists")
                
                # Enhanced debugging for splitting issues
                elif (result.get('status') == 'success' and 
                      result.get('text_blocks_count', 0) > 0 and 
                      result.get('extraction_statistics', {}).get('identified_key_value_pairs', 0) == 0):
                    
                    print("\n🔧 Enhanced Debugging Recommendations:")
                    print("   • Server console should show '=== STARTING TEXT BLOCK SPLITTING ==='")
                    print("   • Look for manual split messages like '🎯 MANUAL TEST: Splitting...'")
                    print("   • If no splitting messages appear, the extraction pipeline may not be called")
                    print("   • Check server restart - new debug code may not be loaded")
                    
                    debug_choice = input("\n   Want to see server logs tip? (y/n): ").strip().lower()
                    if debug_choice == 'y':
                        print("\n📋 To see server debug output:")
                        print("   1. Check your server terminal/console window")
                        print("   2. Look for detailed splitting output with exact text matches")
                        print("   3. If using Docker: docker logs <container_name>")
                        print("   4. If no debug output appears, restart the server to load new code")
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
            print("\n🔧 Debug Mode Information:")
            print("="*50)
            print("The document scanner now includes enhanced debugging features:")
            print("\n📊 Server-Side Debug Output:")
            print("   • Text block splitting process is logged to server console")
            print("   • Look for '=== STARTING TEXT BLOCK SPLITTING ===' messages")
            print("   • Manual splitting attempts show '🎯 MANUAL TEST' messages")
            print("   • Each splitting strategy is logged with success/failure")
            
            print("\n🔍 Client-Side Enhancements:")
            print("   • Split blocks are marked with 🔄 indicators")
            print("   • Extraction methods show how pairs were found")
            print("   • Target pattern analysis shows what text was detected")
            print("   • Enhanced statistics show splitting success rate")
            
            print("\n🛠️  Troubleshooting:")
            print("   • If no splitting output appears, restart the server")
            print("   • Check server console/terminal for detailed logs")
            print("   • Confidence values 0.95/0.90 indicate split blocks")
            print("   • Intelligent splitting uses NLP for dynamic pattern recognition")
            
            print("\n💡 Expected Behavior:")
            print("   • Combined text blocks like 'Field Name Value' should split into 'Field Name' + 'Value'")
            print("   • NLP analysis identifies form field labels vs values using POS tagging") 
            print("   • Spatial proximity and confidence scoring pair split blocks correctly")
            print("   • Dynamic pattern learning adapts to any document type automatically")
            
        elif choice == "5":
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
