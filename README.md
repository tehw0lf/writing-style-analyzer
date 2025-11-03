# Writing Style Analyzer

A local writing style analyzer that uses Large Language Models (LLMs) to analyze and profile writing styles in German and English text. This tool runs completely locally without external API calls.

## Features

- **Local LLM Integration**: Uses HuggingFace Transformers or llama.cpp with GGUF models
- **Multilingual Support**: Optimized for German and English text analysis
- **Comprehensive Analysis**:
  - Sentence and paragraph structure
  - Lexical diversity metrics
  - Language-specific features (German formality, compound words, etc.)
  - Common phrases and vocabulary patterns
  - Tone and formality detection
- **Dual-Format Output**: Generates both JSON (for analysis) and Markdown (for AI agents)
- **Profile Generation**: Creates detailed profiles for different writing contexts
- **No External Dependencies**: Runs completely offline using local models

## Project Structure

```
writing-style-analyzer/
├── analyze.py                      # Main profile generation tool ⭐
├── german_academic_analyzer.py     # Universal German text analysis library ⭐⭐
├── pyproject.toml                  # UV project configuration
├── config.yaml                     # Configuration file
├── texts/                          # Input directory for text samples
├── profiles/                       # Output directory for generated profiles
├── user-profiles/                  # V2 validated profiles and documentation
│   ├── profiles/                   # Validated academic profiles (default, excellence)
│   ├── test-prompts/               # Test validation framework
│   ├── validate_test*.py           # Test validation scripts (⚠️ user-specific)
│   └── *.md                        # Comprehensive usage guides
├── SCRIPTS_README.md               # Guide to all analysis scripts ⭐
└── README.md                       # This file
```

**Key Files for Other Users:**
- `analyze.py` - Create your own writing profile ✅
- `german_academic_analyzer.py` - Universal German text analyzer ✅
- `user-profiles/validate_test*.py` - ⚠️ SKIP these (hardcoded to original author)

**See `SCRIPTS_README.md` for detailed explanation of each script!**

## Installation

### Prerequisites

- Python 3.10 or higher
- [UV package manager](https://github.com/astral-sh/uv)

### Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup Project

```bash
# Navigate to project directory
cd writing-style-analyzer

# Create virtual environment and install dependencies
uv venv
uv sync

# Or if using pip:
uv pip install -e .
```

### Model Setup

The analyzer uses HuggingFace models by default. On first run, the model will be downloaded automatically (~3-7GB depending on model choice).

**Recommended Models for German/English:**

1. **Qwen/Qwen2.5-3B-Instruct** (Default, excellent multilingual support)
2. **meta-llama/Llama-3.2-3B-Instruct** (Good multilingual performance)
3. **mistralai/Mistral-7B-Instruct-v0.2** (Larger, better quality, needs more resources)

Configure your preferred model in `config.yaml`:

```yaml
model:
  type: "transformers"
  name: "Qwen/Qwen2.5-3B-Instruct"
  device: "auto"  # auto-detects GPU/CPU
```

## Configuration

Edit `config.yaml` to customize:

- **Model settings**: Model type, name, device, parameters
- **Analysis settings**: Chunk size, languages, detail level
- **File processing**: Extensions, encoding, ignore patterns
- **Output settings**: JSON formatting, example inclusion

See the `config.yaml` file for detailed comments on all options.

## Usage

### Basic Usage

```bash
# Analyze blog posts
uv run analyze.py --input texts/blog --output profiles/blog-profile.json --profile-type blog

# Analyze social media content
uv run analyze.py --input texts/social --output profiles/social-profile.json --profile-type social

# Use custom config
uv run analyze.py --input texts/blog --output profiles/custom.json --config my-config.yaml
```

### Command-Line Options

```
Options:
  --input, -i       Input directory containing text files (required)
  --output, -o      Output path for profile JSON (required)
  --profile-type, -t Profile type name (default: general)
  --config, -c      Path to config file (default: config.yaml)
  --help, -h        Show help message
```

### Example Workflow

1. **Collect your text samples**:
   ```bash
   mkdir -p texts/blog
   # Copy your writing samples (.txt, .md, .pdf, .docx, .odt)
   ```

2. **Run analysis**:
   ```bash
   uv run analyze.py --input texts/blog --output profiles/my-blog.json --profile-type tech-blog
   ```

3. **Review the profile**:
   ```bash
   cat profiles/my-blog.json
   ```

## Profile Output Format

The analyzer generates **two files** for each profile:

1. **JSON file** (`profile-name.json`): Complete analysis data, metrics, and metadata
2. **Markdown file** (`profile-name.md`): AI-friendly instructions for writing guidance

### JSON Profile Structure

The JSON profile contains the following structure:

```json
{
  "profile_name": "tech-blog",
  "created_at": "2025-10-26T12:34:56.789",
  "analyzed_files": 15,
  "primary_language": "de",
  "languages_detected": ["de", "en"],
  "metrics": {
    "avg_sentence_length": 18.5,
    "avg_paragraph_length": 3.2,
    "lexical_diversity": 0.73,
    "total_words": 5420,
    "total_sentences": 293
  },
  "style_characteristics": {
    "tone": "friendly-informative, conversational",
    "formality": "casual-professional",
    "typical_elements": [
      "Uses 'du' form (German informal you)",
      "Starts with questions or scenarios",
      "Short paragraphs (2-4 sentences)"
    ],
    "structural_patterns": [
      "Question-led openings",
      "Code examples embedded",
      "Summary conclusions"
    ]
  },
  "vocabulary": {
    "common_phrases": [
      "im grunde",
      "tatsächlich",
      "aber",
      "eigentlich"
    ],
    "characteristics": "Mix of German and English technical terms"
  },
  "german_features": {
    "formality": "informal (du-form)",
    "has_compound_words": true,
    "compound_word_examples": ["softwareentwicklung", "datenbankverbindung"],
    "uses_umlauts": true
  },
  "avoid": [
    "Marketing language",
    "Passive voice",
    "Overly formal structures"
  ]
}
```

### Markdown Profile Format

The markdown file provides AI-friendly instructions:

```markdown
# Profile Name Writing Style Profile

## Quick Instructions
Write in this style using these characteristics:

### Voice & Structure
- **Passive voice:** 45%
- **Sentence length:** ~20 words average
- **Lexical diversity:** 0.35

### Transition Words
**Contrastive:**
- Use: jedoch, allerdings, dennoch
- **Target:** ~25 uses per document

### Style Signature
- **Tone:** Professional and technical
- **Formality:** Formal

### What to Avoid
- Colloquial language
- Personal opinions without evidence
```

## Using Profiles with AI Assistants

Once you've generated a profile, you can use it to guide AI assistants when writing new text.

### Method 1: Upload Profile as Project Knowledge (Recommended)

**Best for:** Regular use, convenience

1. Create a project in your AI platform (Claude Desktop, ChatGPT, etc.)
2. Upload the generated `.md` profile file as project knowledge
3. Reference it in your prompts

**Example:**
```
Write a 500-word paragraph about [TOPIC] using my writing style from the profile.
```

### Method 2: Paste Profile in Each Conversation

**Best for:** One-off use, testing different profiles

1. Open the generated `.md` profile file
2. Copy the entire content
3. Paste it into your AI conversation
4. Follow with your writing request

**Example:**
```
[Paste full profile content]

Based on this writing style profile, write about [TOPIC]...
```

### Method 3: Reference Specific Metrics

**Best for:** Fine-tuning specific aspects

Extract key metrics from your profile and reference them:
```
Write a paragraph with:
- Average sentence length: ~[X] words
- Passive voice ratio: ~[Y]%
- Use transitions from categories: [list]
```

### Integration Options

**MCP Memory** (if available):
Store profiles in memory for later retrieval

**File Attachment** (if available):
Attach the `.md` file directly to conversations

**See `user-profiles/` directory for example usage guides and validation results.**

## German Text Analysis Features

The analyzer includes specialized support for German language features:

### Formality Detection
- **du-form** (informal): du, dich, dir, dein
- **Sie-form** (formal): Sie, Ihnen, Ihr

### Compound Words
Detects long German compound words (e.g., "Softwareentwicklungsumgebung")

### Umlauts & Special Characters
Full UTF-8 support for ä, ö, ü, ß

### Sentence Structure
Adapts to typically longer German sentences compared to English

## Hardware Requirements

### Minimum
- **CPU**: Modern x86_64 processor
- **RAM**: 8GB (for 3B parameter models)
- **Storage**: 10GB free space

### Recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA support)
- **RAM**: 16GB
- **Storage**: 20GB free space

### Performance Tips

1. **Use GPU acceleration** when available:
   ```yaml
   model:
     device: "cuda"  # or "mps" for Apple Silicon
   ```

2. **Use smaller models** for faster analysis:
   - 3B models: Fast, good quality
   - 7B models: Slower, better quality

3. **Adjust chunk size** in config for memory constraints:
   ```yaml
   analysis:
     chunk_size: 4000  # Reduce if running out of memory
   ```

## Troubleshooting

### Out of Memory Errors

**Symptoms**: Process killed or CUDA out of memory

**Solutions**:
1. Use CPU instead of GPU: `device: "cpu"` in config
2. Use smaller model (3B instead of 7B)
3. Reduce chunk_size in config
4. Close other applications

### Model Download Issues

**Symptoms**: Connection timeouts or download failures

**Solutions**:
1. Check internet connection
2. Use HuggingFace mirror if available
3. Manually download model and configure path
4. Try alternative model

### Language Detection Issues

**Symptoms**: Wrong language detected

**Solutions**:
1. Ensure text files have sufficient content (>100 words)
2. Check UTF-8 encoding is correct
3. Mixed-language texts may show "en" as primary if English dominates

### Slow Performance

**Symptoms**: Analysis takes very long

**Solutions**:
1. Enable GPU acceleration in config
2. Use smaller/faster model
3. Reduce number of input files
4. Increase chunk_size for batch processing

## Advanced Usage

### Using GGUF Models (llama.cpp)

For potentially better performance with quantized models:

1. **Install llama-cpp-python**:
   ```bash
   uv pip install llama-cpp-python
   ```

2. **Download a GGUF model** (e.g., from HuggingFace)

3. **Configure**:
   ```yaml
   model:
     type: "llama-cpp"
     path: "/path/to/model.gguf"
   ```

### Batch Processing Multiple Directories

```bash
#!/bin/bash
for dir in texts/*/; do
  profile_name=$(basename "$dir")
  uv run analyze.py --input "$dir" --output "profiles/${profile_name}.json" --profile-type "$profile_name"
done
```

### Custom Analysis Parameters

Create multiple config files for different use cases:

```bash
# Quick analysis (lower quality, faster)
uv run analyze.py --input texts/blog --output profiles/quick.json --config config-fast.yaml

# Detailed analysis (higher quality, slower)
uv run analyze.py --input texts/blog --output profiles/detailed.json --config config-detailed.yaml
```

## Development

### Testing

The project includes a comprehensive automated test suite with 49 tests covering profile validation, analysis functions, and regression testing.

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=term-missing

# Run specific test categories
uv run pytest tests/ -m profile      # Profile validation
uv run pytest tests/ -m analysis     # Analysis functions
uv run pytest tests/ -m regression   # Regression tests
```

**Privacy-First Design:** All tests use synthetic data only. Your personal profiles and texts remain private (gitignored).

See [tests/README.md](tests/README.md) for complete test suite documentation.

### Project Dependencies

Core dependencies:
- `transformers`: HuggingFace model support
- `torch`: PyTorch for model inference
- `pyyaml`: Configuration file parsing
- `langdetect`: Language detection
- `tqdm`: Progress bars
- `pypdf`: PDF text extraction
- `python-docx`: Microsoft Word (.docx) text extraction
- `odfpy`: LibreOffice Writer (.odt) text extraction

Optional:
- `llama-cpp-python`: GGUF model support

Development:
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `black`: Code formatting
- `ruff`: Linting

### Code Structure

- **TextProcessor**: Text analysis and metric calculation
- **LLMAnalyzer**: LLM integration and style analysis
- **WritingStyleAnalyzer**: Main orchestrator
- **Configuration**: YAML-based configuration management

## Model Recommendations

### For German + English (Bilingual)

| Model | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| Qwen2.5-3B-Instruct | 3B | ⭐⭐⭐⭐ | ⚡⚡⚡ | Best balance, default |
| Llama-3.2-3B | 3B | ⭐⭐⭐ | ⚡⚡⚡ | Good alternative |
| Mistral-7B-Instruct | 7B | ⭐⭐⭐⭐⭐ | ⚡⚡ | Best quality, slower |

### For German Primary

Qwen2.5 series has excellent German support and is recommended for German-heavy content.

## License

This project is provided as-is for personal and educational use.

## Contributing

Contributions welcome! Areas for improvement:
- Additional language support
- Profile comparison tools
- Statistical validation metrics
- Web interface

## Support

For issues and questions:
1. Check this README and `config.yaml` comments
2. Review logs in `analyzer.log`
3. Check HuggingFace model documentation
4. Verify Python and dependency versions

## Example Profiles

This repository includes example profile documentation in the `user-profiles/` directory (gitignored for privacy). This shows how to organize your personal writing style profiles and documentation.

### Profile Organization

**Project-level (this directory):**
- Tool documentation (README, QUICKSTART, CLAUDE.md)
- Example texts for testing
- Analyzer source code

**User-level (`user-profiles/` - gitignored):**
- Your analyzed writing style profiles
- Profile usage guides
- Comparison and test documentation

### Creating Your Own Profiles

The `user-profiles/` directory is where you'll store your generated profiles and documentation. This directory is **gitignored** to protect your privacy.

**To create your first profile:**

1. **Collect text samples** (10-20 files, 5000+ words total)
   ```bash
   mkdir -p texts/my-writing
   # Copy your .txt, .md, .pdf, .docx files here
   ```

2. **Run the analyzer:**
   ```bash
   uv run analyze.py --input texts/my-writing --output profiles/my-style.json --profile-type my-style
   ```

3. **Review the output:**
   - `profiles/my-style.json` - Complete analysis data
   - `profiles/my-style.md` - AI-friendly profile for guidance

4. **Use with AI assistants:**
   - Upload the `.md` file to your AI platform
   - Reference it when asking for text generation

**Profile Organization:**

We recommend creating a `user-profiles/` directory structure:
```
user-profiles/
├── profiles/           # Your generated profiles
│   ├── academic.json
│   ├── academic.md
│   ├── blog.json
│   └── blog.md
└── README.md          # Your personal usage notes
```

**Example profiles are available in the repository's issue tracker for reference, but your profiles will be unique to your writing style.**

## Changelog

### v0.5.0 (2025-10-27)
- **Hybrid Pattern Discovery System:** Major upgrade to profile generation
  - Combines authoritative patterns from Duden/academic style guides with LLM-discovered patterns
  - Generates profiles with 3-4x more linguistic patterns than basic analysis
  - New transition categories: conditional, clarifying, concessive
  - Improved passive voice accuracy and argumentation detection
- **Dual-Format Output:** Profiles now generated in both JSON and Markdown
  - JSON for analysis and metrics
  - Markdown for AI assistant integration
- **Comprehensive Documentation:** Profile usage guides and validation framework
  - Profile creation guide
  - AI integration best practices
  - Validation test framework

### v0.4.0 (2025-10-27)
- **Documentation restructuring:** Separated project and user-specific documentation
  - Created `user-profiles/` directory for personal profiles (gitignored)
  - Moved profile-specific documentation to `user-profiles/`
  - Updated .gitignore to protect user privacy
- **Profile management improvements:** Simplified profile organization
  - Clearer naming conventions for generated profiles
  - Profile archiving and versioning support
  - Validation framework for testing profile quality
- **Empirical validation:** Testing framework confirms profile quality and distinctiveness

### v0.3.0 (2025-10-27)
- Added LaTeX (.tex) file support with pylatexenc
- Replaced pypdf with pdfplumber for better PDF text extraction
- Added comprehensive linguistic analysis:
  - Voice analysis (passive vs active)
  - Transition word analysis (5 categories)
  - Sentence complexity metrics
  - Rhetorical device detection
- Improved content filtering (code/formula/reference detection)
- Enhanced phrase extraction with stopword filtering
- Robust JSON parsing with retry logic
- Created pre-analyzed academic profiles with detailed documentation

### v0.2.0 (2025-10-26)
- Improved German language support
- Added profile merging capabilities
- Enhanced error handling

### v0.1.0 (2025-10-26)
- Initial release
- German and English support
- HuggingFace Transformers integration
- Basic profile generation
- JSON output format
