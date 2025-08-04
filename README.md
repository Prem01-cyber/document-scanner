# 🧠 Enhanced Document Scanner

**Advanced document processing with hybrid quality assessment and intelligent key-value extraction**

## 🚀 Quick Start

For complete setup and running instructions, see: **[HOW_TO_RUN.md](HOW_TO_RUN.md)**

```bash
# Quick setup
python scripts/quick_setup.py

# Launch enhanced UI
python enhanced_gradio_app.py

# Open browser: http://localhost:7861
```

## 📁 Project Structure

The project is now organized into logical directories:
- **`src/`** - Core processing modules
- **`quality/`** - Hybrid quality assessment system  
- **`demos/`** - Demo scripts and examples
- **`tests/`** - Test suites
- **`docs/`** - Documentation
- **`scripts/`** - Setup and utility scripts
- **`config/`** - Configuration files
- **`deploy/`** - Deployment scripts and Docker files

## 🔁 End-to-End Pipeline Overview

### ➤ **You give an image (`.jpg`, `.png`) as input**

This can happen via:

* CLI
* API (`/scan-document`, `/scan-adaptive-only`, etc.)
* Gradio (if you decide to build a UI)

---

## 📷 STEP 1: **Image Conversion**

```python
nparr = np.frombuffer(image_bytes, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```

* Converts raw bytes to an OpenCV image.
* This is used by both **quality assessment** and **OCR**.

---

## 🧮 STEP 2: **Quality Assessment with Learning**

### Module: `AdaptiveDocumentQualityChecker`

#### Actions:

1. **Calculate adaptive thresholds**:

   * From `adaptive_config`, it fetches current values like:

     * `blur_threshold`, `dark_threshold`, `bright_threshold`, `skew_tolerance`, etc.
2. **Analyzes**:

   * **Blurriness** using Laplacian
   * **Brightness percentiles** using histograms
   * **Edge density**, **document area ratio**, **skew** via HoughLines
3. **Returns:**

   ```json
   {
     "needs_rescan": false,
     "confidence": 0.86,
     "issues": [],
     "adaptive_thresholds": { ... }
   }
   ```

#### ✨ Learning:

If quality results are correlated with successful extraction later,
→ it **adjusts the thresholds** using:

```python
adaptive_config.update_learning("quality_thresholds", "blur_threshold", new_value)
```

---

## 🧠 STEP 3: **OCR with Google Vision**

### Module: `GoogleOCRProcessor`

* Calls `document_text_detection` via `vision.ImageAnnotatorClient`
* Extracts:

  * Text blocks
  * Their **bounding boxes**
  * **Confidence scores**

Example output:

```python
[
  TextBlock(
    text="Invoice Number:",
    bbox=BoundingBox(x=120, y=80, width=90, height=25),
    confidence=0.94
  ),
  ...
]
```

---

## 🔍 STEP 4: **Key-Value Extraction**

### Two parallel paths: **Adaptive** and **LLM**

Controlled by the **extraction strategy**:

* `"adaptive_first"` (default)
* `"llm_first"`, `"parallel"`, etc.

---

### 🧩 STEP 4A: Adaptive Key-Value Extraction

#### Module: `AdaptiveKeyValueExtractor`

1. **Identify key candidates** using:

   * Structural: ends with colon, contains “ID”, etc.
   * Spatial: left aligned, in same line
   * Semantic: spaCy-based entity similarity
   * Pattern match: learned examples like “Invoice No.”, “Ref #”

2. **Match with nearby value blocks** based on:

   * **Distance**
   * **Alignment**
   * **Semantic match**

3. **Compute confidence** per pair:

   ```python
   KeyValuePair(
     key="Invoice Number",
     value="INV-782394",
     confidence=0.81,
     extraction_method="adaptive_spatial"
   )
   ```

#### ✨ Learning:

Each successful extraction is stored in:

```python
self.successful_extractions.append(KeyValuePair(...))
```

From this, it updates:

* Boosts for colon/suffix detection
* Distance thresholds
* Key length confidence patterns

Stored in `adaptive_config.config["extraction_confidence"]`.

---

### 🧠 STEP 4B: LLM-Based Extraction

#### Module: `LLMKeyValueExtractor`

* Takes `raw_text` from OCR
* Sends it to:

  * **OpenAI GPT-4o-mini**
  * Or fallback (e.g., Ollama, Azure)
* Returns key-value pairs (structured JSON)

Example:

```json
[
  {
    "key": "Invoice Number",
    "value": "INV-782394",
    "confidence": 0.9,
    "extraction_method": "llm_extraction",
    "llm_provider": "openai"
  }
]
```

#### ✨ Learning:

Tracks:

* How often fallback was used
* How confident LLMs are across document types
* When LLM outperforms adaptive → logs it for strategic improvement

---

## 🧠 STEP 5: Hybrid Coordination & Decision

### Module: `HybridKeyValueExtractor`

Based on strategy (`ADAPTIVE_FIRST`, `CONFIDENCE_BASED`, etc.):

* **Uses Adaptive if good enough**, else **LLM**
* Or **runs both** and compares/merges
* Tracks:

  * `adaptive_confidence`, `llm_confidence`
  * `fallback_used`, `extraction_time_seconds`

Returns a unified result.

---

## 📈 STEP 6: Learning from the Run

### Triggered in:

* `HybridKeyValueExtractor._learn_from_extraction(...)`
* `AdaptiveDocumentProcessor._learn_from_processing_session(...)`

#### Updates:

* Adaptive thresholds
* Confidence boosts
* Strategy preference
* Pattern confidence
* History in `adaptive_config.processing_history`

Example:

```json
{
  "average_confidence": 0.83,
  "quality_assessment": { ... },
  "extraction_statistics": {
    "total_pairs": 10,
    "high_confidence_pairs": 8
  }
}
```

---

## 🧠 STEP 7: Updating `adaptive_config`

This is the **brain** storing everything adaptive.

For example:

```json
"colon_terminator_boost": {
  "default": 0.2,
  "learned_values": [0.25, 0.3, 0.35],
  "confidence_weight": 0.6
}
```

It uses moving averages, decay, and sample thresholds to update values over time:

```python
def get_adaptive_value(category, param):
    return weighted_avg(default, learned_values, confidence_weight)
```

---

## 🧪 Optional: Run `demo_adaptive_learning.py`

To simulate this whole pipeline with random but realistic variations.

* It prints:

  * Confidence evolution
  * Threshold learning
  * Parameter trend

---

## 📦 In Summary

| Stage              | Learns From                     | Adaptively Changes                  |
| ------------------ | ------------------------------- | ----------------------------------- |
| Quality Check      | Blur/brightness success rates   | Thresholds for blur/brightness/skew |
| Key Detection      | Successful keys in real docs    | Colon/suffix boosts, spatial rules  |
| Value Matching     | Good pairings                   | Distance thresholds, heuristics     |
| LLM Usage          | Fallback frequency/confidence   | Provider preferences, model tuning  |
| Strategy Switching | Confidence deltas over sessions | Strategy preference in config       |
