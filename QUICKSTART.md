# Quick Start Guide

Get up and running with the Writing Style Analyzer in 5 minutes!

## Prerequisites

- Python 3.10+
- At least 8GB RAM
- 10GB free disk space
- Internet connection (for initial model download)

## Installation (5 minutes)

### 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Project

```bash
cd writing-style-analyzer

# Create virtual environment and install dependencies
uv venv
uv sync
```

This will download and install all required packages. The first run will also download the LLM model (~3-7GB).

## First Analysis (2 minutes)

### 1. Add Your Text Files

```bash
# Create a directory for your texts
mkdir -p texts/my-blog

# Copy your writing samples (.txt, .md, .pdf, .docx, .odt)
cp /path/to/your/texts/* texts/my-blog/
```

Or use the provided examples:

```bash
# Examples are already in texts/examples/
ls texts/examples/blog/
ls texts/examples/social/
```

### 2. Run Analysis

```bash
# Analyze the example blog texts
uv run analyze.py --input texts/examples/blog --output profiles/blog-example.json --profile-type blog
```

This will:
- Load the LLM model (first run takes longer)
- Analyze all text files in `texts/examples/blog/`
- Generate a profile in `profiles/blog-example.json`

### 3. View Results

```bash
# View the generated profile
cat profiles/blog-example.json

# Or use jq for pretty output
cat profiles/blog-example.json | jq
```

## Common First-Time Issues

### "Out of memory" error

**Solution**: Use CPU instead of GPU in `config.yaml`:

```yaml
model:
  device: "cpu"
```

### Model download fails

**Solution**: Check your internet connection, or download the model manually and configure the path.

### ImportError for torch/transformers

**Solution**: Run `uv sync` again to ensure all dependencies are installed.

## Next Steps

### Customize Your Analysis

Edit `config.yaml` to:
- Change the model (try different sizes/providers)
- Adjust analysis parameters
- Configure output format

### Analyze Different Text Types

```bash
# Example social media posts
uv run analyze.py --input texts/examples/social --output profiles/social-example.json --profile-type social

# Your own blog texts
uv run analyze.py --input texts/my-blog --output profiles/my-blog.json --profile-type blog

# Your own texts
uv run analyze.py --input texts/my-writing --output profiles/my-style.json --profile-type personal
```

### Use the Profile

The generated JSON profile contains:
- Quantitative metrics (sentence length, vocabulary richness)
- Qualitative characteristics (tone, formality)
- Common phrases and patterns
- German-specific features (if applicable)

Use this profile to:
- Guide AI writing assistants
- Maintain consistent style across content
- Analyze style evolution over time
- Compare different writing contexts

## Performance Tips

### Speed Up Analysis

1. **Use GPU**: Set `device: "cuda"` in config (requires NVIDIA GPU)
2. **Smaller model**: Use 3B models instead of 7B
3. **Fewer files**: Start with a subset of your texts

### Better Quality

1. **More text samples**: 10+ files recommended
2. **Longer texts**: Each file should have 200+ words
3. **Consistent style**: Use texts from the same context/period

## Getting Help

Check the main [README.md](README.md) for:
- Detailed configuration options
- Troubleshooting guide
- Advanced usage examples
- Model recommendations

## Example Workflow

```bash
# 1. Collect texts from your blog
mkdir -p texts/tech-blog
cp ~/blog-posts/*.md texts/tech-blog/

# 2. Run analysis
uv run analyze.py \
  --input texts/tech-blog \
  --output profiles/tech-blog-2024.json \
  --profile-type tech-blog

# 3. Review the profile
cat profiles/tech-blog-2024.json | jq '.style_characteristics'

# 4. Use in your AI prompts
# "Please write in this style: [paste relevant profile sections]"
```

## What's Analyzed?

The analyzer examines:

- ✅ **Structure**: Sentence/paragraph length patterns
- ✅ **Vocabulary**: Word choice, technical terms, common phrases
- ✅ **Tone**: Formal vs casual, friendly vs neutral
- ✅ **Language features**: German formality (du/Sie), compound words
- ✅ **Style elements**: Use of questions, examples, metaphors

Happy analyzing! 🎉
