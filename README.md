# Nano Banana MCP

A standalone MCP server for AI image generation using Google's Nano Banana (Gemini) models.

## Features

- **`generate_image`** - Generate images and return as base64
- **`generate_image_to_file`** - Generate and save directly to a file path
- **`list_styles`** - List available style presets

## Models

| Model | ID | Notes |
|-------|-----|-------|
| Nano Banana | `gemini-2.0-flash-exp` | Free tier available (10 req/min, 1500/day) |
| Nano Banana Pro | `imagen-3.0-generate-002` | Higher quality, requires billing |

## Quick Start

### 1. Get a Gemini API Key

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Install

```bash
cd nano-banana-mcp
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

### 3. Configure Claude Code

Add to your Claude Code MCP settings (`~/.claude/settings.json` or VS Code settings):

```json
{
  "mcpServers": {
    "nano-banana": {
      "command": "python",
      "args": ["/path/to/nano-banana-mcp/server.py"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Or using uv:
```json
{
  "mcpServers": {
    "nano-banana": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/nano-banana-mcp", "python", "server.py"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Usage Examples

### Generate image for Slidev presentation

```
Generate a hero image for my presentation about AI:
- prompt: "Abstract neural network visualization with flowing data streams, blue and purple gradients on dark background"
- output_path: "/path/to/presentation/public/hero.png"
- style_preset: "tech"
- aspect_ratio: "16:9"
```

### Generate with style preset

```
Generate an image:
- prompt: "Team collaboration concept showing diverse hands building something together"
- style_preset: "corporate"
- aspect_ratio: "16:9"
```

## Style Presets

| Preset | Best For |
|--------|----------|
| `corporate` | Business presentations, professional documents |
| `minimal` | Clean slides, modern designs, tech presentations |
| `vibrant` | Marketing materials, creative presentations |
| `tech` | Technology topics, software presentations, AI/ML content |
| `abstract` | Title slides, section dividers, artistic backgrounds |
| `photographic` | Product showcases, realistic visuals |
| `illustration` | Educational content, storytelling |

## Aspect Ratios

| Ratio | Use Case |
|-------|----------|
| `16:9` | Presentations, slides (default) |
| `3:2` | Photography-style images |
| `1:1` | Social media, icons |
| `2:3` | Portrait/vertical |
| `9:16` | Mobile/stories |

## Integration with Slidev

This MCP works great with the [Slidev skill](https://github.com/jesper/slidev-skill). Generate images to your `public/` folder, then reference them in slides:

```markdown
---
layout: image-right
image: /hero.png
---

# My Title

Content here
```

## License

MIT
