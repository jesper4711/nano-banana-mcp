#!/usr/bin/env python3
"""
Nano Banana MCP Server

A standalone MCP server for AI image generation using Google's Gemini API
(Nano Banana / Nano Banana Pro models).

Supports both:
- Google AI Studio (via GEMINI_API_KEY)
- Vertex AI (via GOOGLE_GENAI_USE_VERTEXAI=True + gcloud auth)

Provides tools for generating images that can be used in presentations,
documents, or any other context requiring AI-generated visuals.
"""

import base64
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# Initialize the MCP server
mcp = FastMCP("nano_banana_mcp")

# Initialize Gemini client
# The client auto-detects configuration from environment variables:
# - GEMINI_API_KEY: Use Google AI Studio
# - GOOGLE_GENAI_USE_VERTEXAI=True + GOOGLE_CLOUD_PROJECT: Use Vertex AI
if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
    # Vertex AI mode - uses Application Default Credentials (gcloud auth)
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )
else:
    # AI Studio mode - uses API key
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class AspectRatio(str, Enum):
    """Supported aspect ratios for image generation."""
    LANDSCAPE_16_9 = "16:9"
    LANDSCAPE_3_2 = "3:2"
    SQUARE = "1:1"
    PORTRAIT_2_3 = "2:3"
    PORTRAIT_9_16 = "9:16"


class StylePreset(str, Enum):
    """Style presets for consistent image generation."""
    CORPORATE = "corporate"
    MINIMAL = "minimal"
    VIBRANT = "vibrant"
    TECH = "tech"
    ABSTRACT = "abstract"
    PHOTOGRAPHIC = "photographic"
    ILLUSTRATION = "illustration"


class ModelChoice(str, Enum):
    """Available Nano Banana models."""
    NANO_BANANA = "gemini-2.0-flash-exp"  # Free tier, faster
    NANO_BANANA_PRO = "imagen-3.0-generate-002"  # Higher quality, requires billing


# Style preset prompts that get appended to user prompts
STYLE_MODIFIERS = {
    StylePreset.CORPORATE: "professional corporate aesthetic, clean and polished, business-appropriate",
    StylePreset.MINIMAL: "minimalist design, clean lines, simple composition, plenty of negative space",
    StylePreset.VIBRANT: "vibrant colors, dynamic composition, energetic and bold",
    StylePreset.TECH: "futuristic tech aesthetic, digital elements, modern and innovative",
    StylePreset.ABSTRACT: "abstract artistic style, geometric shapes, creative interpretation",
    StylePreset.PHOTOGRAPHIC: "photorealistic, high quality photography style, natural lighting",
    StylePreset.ILLUSTRATION: "illustrated style, artistic rendering, hand-drawn aesthetic",
}


class GenerateImageInput(BaseModel):
    """Input model for image generation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        ...,
        description="Detailed prompt describing the image to generate. Be specific about subject, style, colors, and composition.",
        min_length=10,
        max_length=2000
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE_16_9,
        description="Aspect ratio for the generated image. Use 16:9 for presentations/slides."
    )
    style_preset: Optional[StylePreset] = Field(
        default=None,
        description="Optional style preset to apply. Adds style-specific modifiers to your prompt."
    )
    model: ModelChoice = Field(
        default=ModelChoice.NANO_BANANA,
        description="Model to use. 'gemini-2.0-flash-exp' (free tier) or 'imagen-3.0-generate-002' (higher quality, requires billing)"
    )
    negative_prompt: Optional[str] = Field(
        default="text, words, letters, watermark, signature, blurry, low quality",
        description="Things to avoid in the generated image."
    )


class GenerateImageToFileInput(BaseModel):
    """Input model for generating image and saving to file."""
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        ...,
        description="Detailed prompt describing the image to generate.",
        min_length=10,
        max_length=2000
    )
    output_path: str = Field(
        ...,
        description="Full path where the image should be saved (e.g., '/path/to/public/hero.png')"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE_16_9,
        description="Aspect ratio for the generated image."
    )
    style_preset: Optional[StylePreset] = Field(
        default=None,
        description="Optional style preset to apply."
    )
    model: ModelChoice = Field(
        default=ModelChoice.NANO_BANANA,
        description="Model to use for generation."
    )
    negative_prompt: Optional[str] = Field(
        default="text, words, letters, watermark, signature, blurry, low quality",
        description="Things to avoid in the generated image."
    )


def _build_full_prompt(prompt: str, style_preset: Optional[StylePreset]) -> str:
    """Build the full prompt including style modifiers."""
    full_prompt = prompt
    if style_preset and style_preset in STYLE_MODIFIERS:
        full_prompt = f"{prompt}, {STYLE_MODIFIERS[style_preset]}"
    return full_prompt


async def _generate_image(
    prompt: str,
    aspect_ratio: AspectRatio,
    style_preset: Optional[StylePreset],
    model: ModelChoice,
    negative_prompt: Optional[str]
) -> bytes:
    """Generate an image using the Gemini API."""
    full_prompt = _build_full_prompt(prompt, style_preset)

    if model == ModelChoice.NANO_BANANA:
        # Use Gemini 2.0 Flash for image generation
        response = client.models.generate_content(
            model=model.value,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                # Note: aspect ratio control may be limited in this model
            )
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data

        raise ValueError("No image generated in response")

    else:
        # Use Imagen 3 for higher quality
        response = client.models.generate_images(
            model=model.value,
            prompt=full_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio.value,
                negative_prompt=negative_prompt,
            )
        )

        if response.generated_images:
            return response.generated_images[0].image.image_bytes

        raise ValueError("No image generated in response")


@mcp.tool(
    name="generate_image",
    annotations={
        "title": "Generate AI Image",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def generate_image(params: GenerateImageInput) -> str:
    """Generate an AI image using Google's Nano Banana (Gemini) model.

    Returns the image as a base64-encoded string that can be embedded directly
    or saved to a file. Use this for flexible image generation where you want
    to handle the output yourself.

    For presentations, use style_preset='corporate' or 'minimal' and aspect_ratio='16:9'.

    Args:
        params: Image generation parameters including prompt, aspect ratio, and style.

    Returns:
        JSON string containing:
        - image_base64: Base64-encoded PNG image data
        - prompt_used: The full prompt that was sent to the API
        - model: The model that was used
    """
    import json

    try:
        image_bytes = await _generate_image(
            prompt=params.prompt,
            aspect_ratio=params.aspect_ratio,
            style_preset=params.style_preset,
            model=params.model,
            negative_prompt=params.negative_prompt
        )

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        full_prompt = _build_full_prompt(params.prompt, params.style_preset)

        return json.dumps({
            "success": True,
            "image_base64": image_base64,
            "prompt_used": full_prompt,
            "model": params.model.value,
            "aspect_ratio": params.aspect_ratio.value
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "prompt_used": _build_full_prompt(params.prompt, params.style_preset)
        })


@mcp.tool(
    name="generate_image_to_file",
    annotations={
        "title": "Generate AI Image to File",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def generate_image_to_file(params: GenerateImageToFileInput) -> str:
    """Generate an AI image and save it directly to a file.

    This is a convenience tool for workflows like Slidev presentations where
    you want to save images to a specific location (e.g., public/ folder).

    Example: Generate a hero image for a Slidev presentation:
    - prompt: "Abstract visualization of AI neural networks, blue and purple gradients"
    - output_path: "/path/to/presentation/public/hero.png"
    - style_preset: "tech"
    - aspect_ratio: "16:9"

    Args:
        params: Image generation parameters including prompt and output path.

    Returns:
        JSON string containing:
        - file_path: Path where the image was saved
        - prompt_used: The full prompt that was sent to the API
    """
    import json

    try:
        image_bytes = await _generate_image(
            prompt=params.prompt,
            aspect_ratio=params.aspect_ratio,
            style_preset=params.style_preset,
            model=params.model,
            negative_prompt=params.negative_prompt
        )

        # Ensure directory exists
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the image
        output_path.write_bytes(image_bytes)

        full_prompt = _build_full_prompt(params.prompt, params.style_preset)

        return json.dumps({
            "success": True,
            "file_path": str(output_path.absolute()),
            "prompt_used": full_prompt,
            "model": params.model.value,
            "aspect_ratio": params.aspect_ratio.value
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "output_path": params.output_path
        })


@mcp.tool(
    name="list_styles",
    annotations={
        "title": "List Style Presets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_styles() -> str:
    """List all available style presets and their descriptions.

    Use these presets to ensure consistent styling across multiple images.
    Each preset adds specific style modifiers to your prompt.

    Returns:
        JSON string containing all available style presets with descriptions.
    """
    import json

    styles = []
    for preset in StylePreset:
        styles.append({
            "name": preset.value,
            "description": STYLE_MODIFIERS.get(preset, ""),
            "recommended_for": _get_style_recommendations(preset)
        })

    return json.dumps({
        "styles": styles,
        "usage_tip": "Pass the style name as style_preset parameter to generate_image or generate_image_to_file"
    }, indent=2)


def _get_style_recommendations(preset: StylePreset) -> str:
    """Get usage recommendations for each style preset."""
    recommendations = {
        StylePreset.CORPORATE: "Business presentations, professional documents",
        StylePreset.MINIMAL: "Clean slides, modern designs, tech presentations",
        StylePreset.VIBRANT: "Marketing materials, creative presentations",
        StylePreset.TECH: "Technology topics, software presentations, AI/ML content",
        StylePreset.ABSTRACT: "Title slides, section dividers, artistic backgrounds",
        StylePreset.PHOTOGRAPHIC: "Product showcases, realistic visuals",
        StylePreset.ILLUSTRATION: "Educational content, storytelling, friendly tone",
    }
    return recommendations.get(preset, "General purpose")


if __name__ == "__main__":
    mcp.run()
