# CLAUDE.md

This file provides guidance to Claude Code when working with this project.

## Project Overview

This is a Python-based writing style analyzer that uses local Large Language Models to analyze and profile writing styles in German and English text. It runs completely locally without external API calls.

**Key Technologies:**
- Python 3.11+ with UV for dependency management
- HuggingFace Transformers (default) or llama-cpp-python for LLM integration
- Recommended models: Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct
- Libraries: PyTorch, transformers, langdetect, pyyaml, tqdm

## Recent Milestone: V2 Profile Validation ✅

**Date:** 2025-10-27
**Status:** V2 profiles validated as superior to V1

**Key Achievements:**
- **Hybrid pattern system:** Combines authoritative Duden/academic patterns with LLM discovery
- **3.6x more patterns:** 222-240 transition patterns vs 62-70 in v1
- **3 new categories:** Conditional, clarifying, and concessive transitions
- **+50% better argumentation:** 6 causal transitions vs 4 in v1-generated text
- **Better passive voice:** 30-40% accuracy vs 35-50% in v1
- **Personality preserved:** Text remains authentic and natural

**Current Profiles:**
- `academic-default-v2`: 222 patterns, 5 categories (general academic writing)
- `academic-excellence-v2`: 240 patterns, 8 categories (high-stakes publications)

**Documentation:**
- Validation results: `user-profiles/PROFILE_TEST_V2.md`
- Usage guide: `user-profiles/PROFILE_GUIDE.md`
- Quick reference: `user-profiles/PROFILES_QUICKREF.md`
- V1 archive: `user-profiles/v1-archive/` (deprecated)

## Project Structure

```
writing-style-analyzer/
├── analyze.py              # Main analyzer script (~650 lines)
├── pyproject.toml          # UV project configuration
├── .python-version         # Python 3.11
├── config.yaml             # Configuration (model, analysis params)
├── texts/                  # Input text samples
│   ├── blog/              # Example blog posts
│   └── social/            # Example social media
├── profiles/               # Generated JSON profiles (gitignored)
├── README.md              # Full documentation
├── QUICKSTART.md          # Quick start guide
└── .gitignore            # Ignores cache, models, logs, profiles
```

## Common Development Commands

### Environment Setup

```bash
# Install dependencies
uv venv
uv sync

# Or install in development mode
uv pip install -e .
```

### Running Analysis

```bash
# Basic analysis
uv run analyze.py --input texts/blog --output profiles/blog.json --profile-type blog

# With custom config
uv run analyze.py --input texts/blog --output profiles/blog.json --config custom-config.yaml

# Help
uv run analyze.py --help
```

### Code Quality

```bash
# Format code
uv run black analyze.py

# Lint code
uv run ruff check analyze.py

# Format and lint
uv run black analyze.py && uv run ruff check analyze.py
```

### Testing

```bash
# Run all tests (49 tests, <1 second)
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=term-missing

# Run specific categories
uv run pytest tests/ -m profile      # Profile validation
uv run pytest tests/ -m analysis     # Analysis functions
uv run pytest tests/ -m regression   # Regression tests
```

**Privacy-First:** All tests use synthetic data only (no personal profiles/texts committed).

### Pre-commit Validation

**IMPORTANT**: Before committing, always run:

```bash
uv run black analyze.py && uv run ruff check analyze.py && uv run pytest tests/
```

## Architecture

### Core Components

1. **ConfigurationError, AnalysisError**: Custom exceptions
2. **setup_logging()**: Configure console and file logging
3. **load_config()**: Load YAML configuration
4. **TextProcessor**: Text analysis utilities
   - Language detection (langdetect)
   - Sentence/paragraph splitting
   - Word tokenization
   - Lexical diversity calculation
   - German-specific feature detection (du/Sie, compound words, umlauts)
   - Basic metrics calculation

5. **LLMAnalyzer**: LLM integration
   - Model loading (transformers or llama-cpp)
   - Text generation with configurable parameters
   - Style analysis prompts (German/English)
   - Common phrase extraction
   - Device auto-detection (CUDA/MPS/CPU)

6. **WritingStyleAnalyzer**: Main orchestrator
   - File collection and reading
   - Text aggregation
   - Full analysis pipeline
   - Profile generation and saving

### Data Flow

1. **Input**: Directory of .txt/.md/.pdf/.docx/.odt files
2. **Reading**: UTF-8 text files or document extraction (PDF/DOCX/ODT) with progress tracking
3. **Text Processing**: Metrics calculation, language detection
4. **LLM Analysis**: Style characterization via prompts
5. **Profile Generation**: Structured JSON output
6. **Output**: Pretty-printed JSON profile

### Profile Output Format

```json
{
  "profile_name": "string",
  "created_at": "ISO-8601 timestamp",
  "analyzed_files": int,
  "primary_language": "de|en",
  "languages_detected": ["de", "en"],
  "metrics": {
    "avg_sentence_length": float,
    "avg_paragraph_length": float,
    "lexical_diversity": float,
    "total_words": int,
    "total_sentences": int
  },
  "style_characteristics": {
    "tone": "string",
    "formality": "string",
    "typical_elements": ["string"],
    "structural_patterns": ["string"]
  },
  "vocabulary": {
    "common_phrases": ["string"],
    "characteristics": "string"
  },
  "german_features": {
    "formality": "informal|formal|neutral",
    "has_compound_words": bool,
    "compound_word_examples": ["string"],
    "uses_umlauts": bool
  },
  "avoid": ["string"]
}
```

## Configuration

All configuration via `config.yaml`:

### Model Configuration
- **type**: "transformers" or "llama-cpp"
- **name**: HuggingFace model name (for transformers)
- **path**: Path to GGUF file (for llama-cpp)
- **device**: "auto", "cuda", "mps", or "cpu"
- **temperature**: 0.0-1.0 (default: 0.3)
- **max_tokens**: Generation limit (default: 2048)

### Analysis Configuration
- **chunk_size**: Character limit per chunk (default: 8000)
- **min_sentences**: Minimum sentences required (default: 10)
- **supported_languages**: ["de", "en"]
- **primary_language**: "de" (affects prompts)

### File Configuration
- **extensions**: [".txt", ".md", ".pdf", ".docx", ".odt"]
- **encoding**: "utf-8"
- **recursive**: true
- **ignore_patterns**: Glob patterns to skip

### Output Configuration
- **profiles_dir**: "profiles"
- **pretty_json**: true
- **include_examples**: true (future feature)

## German Language Support

The analyzer includes specialized support for German:

### Features Detected
1. **Formality**: du-form vs Sie-form detection via regex patterns
2. **Compound words**: Long words (>15 chars) likely to be compounds
3. **Umlauts**: Presence of ä, ö, ü, ß
4. **Sentence length**: Adapts to typically longer German sentences

### Prompts
- German primary language → German system prompt
- English primary language → English system prompt
- Prompts instruct LLM to recognize language-specific patterns

## Model Recommendations

### Best for German + English

| Model | Parameters | Quality | Speed | Notes |
|-------|-----------|---------|-------|-------|
| Qwen/Qwen2.5-3B-Instruct | 3B | ⭐⭐⭐⭐ | ⚡⚡⚡ | Default, best balance |
| meta-llama/Llama-3.2-3B-Instruct | 3B | ⭐⭐⭐ | ⚡⚡⚡ | Good alternative |
| mistralai/Mistral-7B-Instruct-v0.2 | 7B | ⭐⭐⭐⭐⭐ | ⚡⚡ | Better quality, slower |

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **3B models**: ~6GB disk space
- **7B models**: ~14GB disk space

## Common Issues & Solutions

### Out of Memory
- Use CPU: `device: "cpu"`
- Smaller model: 3B instead of 7B
- Reduce chunk_size in config
- Close other applications

### Model Download Failures
- Check internet connection
- Try alternative model
- Manually download and configure path
- Use HuggingFace mirrors if available

### Slow Performance
- Enable GPU: `device: "cuda"` (NVIDIA) or `device: "mps"` (Apple Silicon)
- Use smaller model
- Reduce number of input files
- Batch similar analyses

### Language Detection Issues
- Ensure sufficient text (>100 words per file)
- Check UTF-8 encoding
- Mixed-language texts may misdetect if English dominates

## Making Changes

### Adding New Analysis Features

1. Add to **TextProcessor** for quantitative metrics
2. Add to **LLMAnalyzer.analyze_style()** for qualitative analysis
3. Update prompt in `analyze_style()` method
4. Update profile output structure
5. Update README documentation

### Adding New Language Support

1. Add language code to `supported_languages` in config
2. Update **TextProcessor.detect_language()** if needed
3. Add language-specific feature detection (like `detect_german_features()`)
4. Create system prompt in target language in `analyze_style()`
5. Test with sample texts

### Using Different Models

**Transformers**:
```yaml
model:
  type: "transformers"
  name: "your-model-name"
```

**Llama.cpp**:
```bash
uv pip install llama-cpp-python
```
```yaml
model:
  type: "llama-cpp"
  path: "/path/to/model.gguf"
```

### Adjusting Analysis Quality

**Faster, lower quality**:
- Smaller model (3B)
- Lower max_tokens (1024)
- Smaller chunk_size (4000)
- Higher temperature (0.5)

**Slower, higher quality**:
- Larger model (7B)
- Higher max_tokens (4096)
- Larger chunk_size (12000)
- Lower temperature (0.2)

## Development Best Practices

1. **Always test with example texts** before analyzing important data
2. **Monitor memory usage** during analysis
3. **Check logs** (`analyzer.log`) for warnings/errors
4. **Validate UTF-8 encoding** for German texts
5. **Version control profiles** to track style evolution
6. **Use descriptive profile names** (tech-blog-2024, social-Q1, etc.)

## Future Enhancements

### Already Implemented ✅
- [x] **V2 Profile System** - Hybrid pattern discovery (v0.5.0, 2025-10-27)
  - Combines authoritative base patterns with LLM discovery
  - 3.6x more linguistic patterns (222-240 vs 62-70)
  - 3 new transition categories (conditional, clarifying, concessive)
  - Validated as superior to v1 (see `user-profiles/PROFILE_TEST_V2.md`)
- [x] Profile comparison mode - See `user-profiles/v1-archive/PROFILE_COMPARISON_V1.md` (v1)
- [x] Profile merge mode - Created synthetic profiles (v1: deprecated, v2: active)
- [x] LaTeX (.tex) file support - Added in v0.3.0
- [x] PDF text extraction - Improved in v0.3.0 with pdfplumber
- [x] Document formats - Added .docx and .odt support

### Nice-to-Have Features
- [ ] Text validation against profile - Check if new text matches profile metrics
- [ ] Markdown export format - Export profiles as readable markdown
- [ ] Language statistics - Show % German vs English per file
- [ ] Automated tests - Unit tests for text processing and analysis
- [ ] Web interface - Streamlit/Gradio UI for easier use
- [ ] Batch processing scripts - Analyze multiple directories at once
- [ ] Profile visualization - Charts/graphs of metrics
- [ ] Style drift detection - Track style changes over time
- [ ] English profile adaptation - See `user-profiles/ENGLISH_ADAPTATION_CONCEPT.md`

## Integration with AI Assistants

The generated profiles can be used to guide AI writing:

```
System prompt: "Write in the following style: [paste profile sections]"

Or more specifically:
- Tone: [profile.style_characteristics.tone]
- Formality: [profile.style_characteristics.formality]
- Typical elements: [profile.style_characteristics.typical_elements]
- Avoid: [profile.avoid]
```

## Related Documentation

**Project Documentation:**
- **README.md**: Full documentation, setup, usage, troubleshooting, changelog
- **QUICKSTART.md**: 5-minute getting started guide
- **config.yaml**: Inline comments for all configuration options

**Profile Documentation (V2):**
- **user-profiles/PROFILE_TEST_V2.md**: Complete v2 validation testing and results
- **user-profiles/PROFILE_GUIDE.md**: Comprehensive v2 profile usage guide
- **user-profiles/PROFILES_QUICKREF.md**: Quick reference for v2 profiles
- **user-profiles/AI_PROMPTS.md**: Ready-to-use prompt templates with v2 profiles
- **user-profiles/profiles/academic-default.md**: Default v2 profile (222 transitions)
- **user-profiles/profiles/academic-excellence.md**: Excellence v2 profile (240 transitions)

**Archive (V1 - Deprecated):**
- **user-profiles/v1-archive/README.md**: Why v1 was deprecated
- **user-profiles/v1-archive/PROFILE_TEST_V1.md**: Original v1 validation (for comparison)
- **user-profiles/v1-archive/**: All v1 documentation (archived)

## Dependencies

Core:
- `transformers>=4.40.0`: HuggingFace model support
- `torch>=2.0.0`: PyTorch for model inference
- `pyyaml>=6.0`: Config file parsing
- `langdetect>=1.0.9`: Language detection
- `tqdm>=4.66.0`: Progress bars
- `sentencepiece>=0.2.0`: Tokenization
- `accelerate>=0.30.0`: Model acceleration
- `pypdf>=5.1.0`: PDF text extraction
- `python-docx>=1.1.0`: Microsoft Word (.docx) text extraction
- `odfpy>=1.4.1`: LibreOffice Writer (.odt) text extraction

Optional:
- `llama-cpp-python>=0.2.0`: GGUF model support

Dev:
- `pytest>=8.0.0`: Testing (not yet implemented)
- `black>=24.0.0`: Code formatting
- `ruff>=0.4.0`: Linting
