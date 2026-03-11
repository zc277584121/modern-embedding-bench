"""Mock data generators for evaluation tasks.

All tasks can run with mock data for development/testing purposes.
Replace with real data loaders for production evaluation.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class MockTextImagePair:
    """A text-image pair for cross-modal retrieval."""

    text: str
    image_bytes: bytes
    category: str


@dataclass
class MockDrivingScene:
    """A mock autonomous driving scene."""

    caption: str
    image_bytes: bytes
    weather: str
    time_of_day: str
    road_type: str
    objects: list[str]


def generate_random_image(width: int = 224, height: int = 224, seed: int | None = None) -> bytes:
    """Generate a random RGB image as PNG bytes."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_solid_color_image(
    color: tuple[int, int, int], width: int = 224, height: int = 224
) -> bytes:
    """Generate a solid-color image."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =============================================================================
# MRL Stress Test Mock Data
# =============================================================================

MRL_TEST_TEXTS = [
    # Semantically similar pairs
    ("The cat sat on the mat.", "A feline was resting on the rug."),
    ("Machine learning models require large datasets.", "Deep learning needs massive amounts of training data."),
    ("The stock market crashed yesterday.", "Financial markets experienced a sharp decline."),
    ("She enjoys playing the piano.", "She likes performing music on the keyboard."),
    ("The restaurant serves excellent Italian food.", "This place has great pasta and pizza."),
    # Semantically dissimilar pairs
    ("The quick brown fox jumps over the lazy dog.", "Quantum computing enables exponential speedup."),
    ("Beautiful sunset over the ocean.", "Database normalization prevents data redundancy."),
    ("The children played in the park.", "Photosynthesis converts CO2 to oxygen."),
]


def get_mrl_test_data() -> list[tuple[str, str, bool]]:
    """Get MRL stress test data: (text_a, text_b, is_similar)."""
    data = []
    # Similar pairs
    for a, b in MRL_TEST_TEXTS[:5]:
        data.append((a, b, True))
    # Dissimilar pairs
    for a, b in MRL_TEST_TEXTS[5:]:
        data.append((a, b, False))
    return data


# =============================================================================
# Cross-Modal Retrieval Mock Data
# =============================================================================

CROSS_MODAL_CATEGORIES = [
    ("A red apple on a wooden table", (200, 50, 50), "food"),
    ("A blue sky with white clouds", (100, 150, 255), "nature"),
    ("A green forest with tall trees", (50, 180, 50), "nature"),
    ("A yellow taxi on a city street", (240, 220, 50), "urban"),
    ("A black cat sitting on a windowsill", (30, 30, 30), "animal"),
    ("A white snowfield under bright sun", (240, 240, 245), "nature"),
    ("An orange basketball on the court", (255, 140, 0), "sports"),
    ("A purple flower in the garden", (150, 50, 200), "nature"),
    ("A brown wooden door of an old house", (140, 90, 50), "architecture"),
    ("A grey concrete highway in the rain", (130, 130, 130), "urban"),
]


def get_cross_modal_data() -> list[MockTextImagePair]:
    """Get cross-modal text-image pairs for retrieval evaluation."""
    return [
        MockTextImagePair(
            text=text,
            image_bytes=generate_solid_color_image(color),
            category=cat,
        )
        for text, color, cat in CROSS_MODAL_CATEGORIES
    ]


# =============================================================================
# Needle-in-a-Haystack Mock Data
# =============================================================================

HAYSTACK_BASE_TEXT = (
    "This is a general background document about various topics. "
    "It covers history, science, art, and technology in a broad overview. "
    "The purpose of this text is to provide padding content that surrounds "
    "the key information we want to test retrieval against. "
)

NEEDLE_FACTS = [
    "The secret password for the vault is 'BlueDragon42'.",
    "The quarterly revenue was exactly $14.7 million.",
    "The experimental drug showed a 73% efficacy rate in Phase III trials.",
    "Server maintenance is scheduled for March 15th at 2:00 AM UTC.",
    "The recommended dosage is 250mg twice daily with food.",
]

NEEDLE_QUERIES = [
    "What is the vault password?",
    "What was the quarterly revenue?",
    "What was the drug efficacy rate?",
    "When is the server maintenance scheduled?",
    "What is the recommended dosage?",
]


def get_needle_haystack_data(
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Generate needle-in-a-haystack test cases.

    Args:
        haystack_lengths: List of total document lengths in chars (default: [1K, 4K, 8K, 16K, 32K])
        needle_positions: Where to insert needle (0.0=start, 0.5=middle, 1.0=end)

    Returns:
        List of test cases, each with 'document', 'query', 'needle', 'position', 'length'
    """
    if haystack_lengths is None:
        haystack_lengths = [1000, 4000, 8000, 16000, 32000]
    if needle_positions is None:
        needle_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    test_cases = []
    for length in haystack_lengths:
        # Build haystack of the target length
        n_repeats = length // len(HAYSTACK_BASE_TEXT) + 1
        haystack = (HAYSTACK_BASE_TEXT * n_repeats)[:length]

        for pos in needle_positions:
            for needle, query in zip(NEEDLE_FACTS, NEEDLE_QUERIES):
                # Insert needle at the specified position
                insert_idx = int(len(haystack) * pos)
                # Ensure we don't split words
                insert_idx = haystack.rfind(" ", 0, insert_idx) + 1 if insert_idx > 0 else 0
                document = haystack[:insert_idx] + " " + needle + " " + haystack[insert_idx:]

                test_cases.append({
                    "document": document,
                    "query": query,
                    "needle": needle,
                    "position": pos,
                    "length": length,
                })

    return test_cases


# =============================================================================
# Autonomous Driving Mock Data (simulating CoVLA-style data)
# =============================================================================

DRIVING_SCENES = [
    {
        "caption": "A sedan is making a left turn at a busy intersection during daytime. "
        "There are pedestrians on the crosswalk and a traffic light showing green.",
        "weather": "sunny",
        "time": "daytime",
        "road": "intersection",
        "objects": ["sedan", "pedestrian", "traffic_light"],
        "color": (100, 100, 100),
    },
    {
        "caption": "Highway driving at night with rain. The vehicle ahead is a truck "
        "with visible taillights. Lane markings are partially obscured by water.",
        "weather": "rainy",
        "time": "night",
        "road": "highway",
        "objects": ["truck", "lane_marking", "taillight"],
        "color": (30, 30, 50),
    },
    {
        "caption": "A cyclist is crossing the road at an unmarked crossing in a residential area. "
        "There are parked cars on both sides of the street.",
        "weather": "cloudy",
        "time": "daytime",
        "road": "residential",
        "objects": ["cyclist", "parked_car", "crossing"],
        "color": (150, 150, 140),
    },
    {
        "caption": "Construction zone with orange cones and a flagman directing traffic. "
        "The road narrows from two lanes to one. Speed limit reduced to 25 mph.",
        "weather": "sunny",
        "time": "daytime",
        "road": "construction",
        "objects": ["cone", "flagman", "sign"],
        "color": (200, 150, 80),
    },
    {
        "caption": "Snow-covered rural road with limited visibility. A deer is standing "
        "near the road edge. No lane markings visible.",
        "weather": "snowy",
        "time": "daytime",
        "road": "rural",
        "objects": ["deer", "snow", "road_edge"],
        "color": (220, 220, 230),
    },
    {
        "caption": "Urban parking lot with multiple vehicles. A pedestrian with a shopping cart "
        "is walking between rows of parked cars.",
        "weather": "sunny",
        "time": "daytime",
        "road": "parking_lot",
        "objects": ["pedestrian", "shopping_cart", "parked_car"],
        "color": (160, 160, 160),
    },
    {
        "caption": "Foggy morning on a two-lane mountain road with sharp curves. "
        "Guardrails are visible. An oncoming vehicle has headlights on.",
        "weather": "foggy",
        "time": "morning",
        "road": "mountain",
        "objects": ["guardrail", "curve", "oncoming_vehicle"],
        "color": (180, 180, 190),
    },
    {
        "caption": "School zone during dismissal time. Multiple children are crossing "
        "with a crossing guard. Speed limit 15 mph sign is visible.",
        "weather": "sunny",
        "time": "afternoon",
        "road": "school_zone",
        "objects": ["children", "crossing_guard", "speed_sign"],
        "color": (170, 170, 130),
    },
]

DRIVING_QUERIES = {
    "weather": {
        "rainy": "Find driving scenes in rainy conditions",
        "snowy": "Find driving scenes with snow on the road",
        "foggy": "Find scenes with fog and limited visibility",
        "sunny": "Find scenes in clear sunny weather",
    },
    "hazard": {
        "pedestrian": "Find scenes where pedestrians are present on or near the road",
        "animal": "Find scenes with animals near the roadway",
        "construction": "Find construction zones or road work areas",
    },
    "road_type": {
        "highway": "Find highway driving scenes",
        "intersection": "Find intersection or junction scenes",
        "residential": "Find residential area driving scenes",
    },
}


def get_driving_scene_data() -> list[MockDrivingScene]:
    """Get mock autonomous driving scenes (simulating CoVLA dataset)."""
    scenes = []
    for i, scene in enumerate(DRIVING_SCENES):
        scenes.append(
            MockDrivingScene(
                caption=scene["caption"],
                image_bytes=generate_solid_color_image(scene["color"], 640, 480),
                weather=scene["weather"],
                time_of_day=scene["time"],
                road_type=scene["road"],
                objects=scene["objects"],
            )
        )
    return scenes


def get_driving_queries() -> dict[str, dict[str, str]]:
    """Get driving scene retrieval queries organized by category."""
    return DRIVING_QUERIES


# =============================================================================
# Chinese Multimodal Mock Data
# =============================================================================

CHINESE_TEXT_IMAGE_PAIRS = [
    ("一只橘色的猫蜷缩在沙发上睡觉", (255, 140, 50), "动物"),
    ("北京天安门广场的夜景灯火辉煌", (200, 50, 50), "风景"),
    ("一碗热气腾腾的兰州拉面", (220, 180, 120), "美食"),
    ("长城在秋天的山脊上蜿蜒延伸", (100, 130, 80), "风景"),
    ("医生在手术室里进行腹腔镜手术", (200, 220, 230), "医疗"),
    ("农民在稻田里弯腰收割水稻", (180, 200, 100), "农业"),
    ("一辆红色跑车停在上海外滩", (220, 50, 50), "城市"),
    ("故宫博物院的金色琉璃瓦屋顶", (220, 180, 50), "建筑"),
    ("太极拳老师在公园里带领学生练习", (100, 180, 100), "文化"),
    ("深圳科技园的现代化写字楼群", (150, 180, 200), "城市"),
]

CHINESE_CROSS_LINGUAL_PAIRS = [
    # (Chinese query, English query — both should retrieve the same image)
    ("一只可爱的小猫", "A cute little kitten"),
    ("雄伟的长城", "The magnificent Great Wall"),
    ("繁忙的城市街道", "A busy city street"),
    ("美丽的日落景色", "A beautiful sunset view"),
    ("传统中国建筑", "Traditional Chinese architecture"),
]

CHINESE_DOCUMENT_QUERIES = [
    # Queries for visual document retrieval (Chinese OCR / charts)
    "这份报告的营收数据是多少",
    "图表中显示的最高温度是多少度",
    "合同中的违约条款内容",
    "菜单上最贵的菜品是什么",
    "这张发票的总金额",
]


def get_chinese_text_image_data() -> list[MockTextImagePair]:
    """Get Chinese text-image pairs."""
    return [
        MockTextImagePair(
            text=text,
            image_bytes=generate_solid_color_image(color),
            category=cat,
        )
        for text, color, cat in CHINESE_TEXT_IMAGE_PAIRS
    ]


def get_chinese_cross_lingual_pairs() -> list[tuple[str, str]]:
    """Get Chinese-English parallel query pairs."""
    return CHINESE_CROSS_LINGUAL_PAIRS


def get_chinese_document_queries() -> list[str]:
    """Get Chinese queries for visual document retrieval."""
    return CHINESE_DOCUMENT_QUERIES
