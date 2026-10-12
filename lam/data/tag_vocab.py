"""Danbooru tag vocabulary -> action-descriptor fields.

This is *dataset annotation*: it labels training images so the LAM Encoder can learn from
them. It is never used at generation time; there the learned encoder reads the descriptor
directly (paper §IV-B), so no tag-translation rules sit between the spec and the U-Net.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# ------------------------------------------------------------------ people / identity

GIRL_COUNT = {"1girl": 1, "2girls": 2, "3girls": 3}
BOY_COUNT = {"1boy": 1, "2boys": 2, "3boys": 3}
CROWD = {"4girls", "5girls", "6+girls", "4boys", "5boys", "6+boys", "crowd", "multiple_others", "everyone"}

AGE_TAGS = {
    "old_man": "elderly", "old_woman": "elderly", "old": "elderly",
    "mature_male": "adult", "mature_female": "adult",
    "child": "child", "toddler": "child", "aged_down": "child",
    "teenage": "teen",
}

# ------------------------------------------------------------------ style attributes

_COLORS = ["blonde", "brown", "black", "blue", "purple", "pink", "white", "grey", "red", "silver",
           "green", "orange", "aqua", "light_brown", "light_blue", "light_purple", "dark_blue",
           "light_green", "dark_green", "dark_red", "yellow"]
HAIR_COLOR = {f"{c}_hair": c.replace("_", " ") for c in _COLORS}
HAIR_COLOR.update({"multicolored_hair": "multicolored", "two-tone_hair": "two-tone",
                   "gradient_hair": "gradient", "streaked_hair": "streaked"})
HAIR_STYLE = {t: t.replace("_", " ") for t in [
    "very_short_hair", "short_hair", "medium_hair", "long_hair", "very_long_hair", "absurdly_long_hair",
    "twintails", "ponytail", "high_ponytail", "side_ponytail", "low_twintails", "braid", "twin_braids",
    "single_braid", "french_braid", "bob_cut", "hair_bun", "double_bun", "single_hair_bun", "hime_cut",
    "ahoge", "messy_hair", "wavy_hair", "curly_hair", "straight_hair", "spiked_hair", "drill_hair",
    "bald", "undercut", "hair_over_one_eye", "blunt_bangs", "swept_bangs", "parted_bangs",
]}
_EYE_COLORS = ["blue", "red", "green", "brown", "purple", "yellow", "pink", "black", "grey", "orange",
               "aqua", "light_blue", "light_brown", "golden", "silver", "white"]
EYE_COLOR = {f"{c}_eyes": c.replace("_", " ") for c in _EYE_COLORS}
EYE_COLOR.update({"heterochromia": "heterochromia"})

EXPRESSION = {t: t.replace("_", " ") for t in [
    "smile", "light_smile", "grin", "smirk", "open_mouth", "closed_mouth", "parted_lips", "laughing",
    "blush", "frown", "crying", "tears", "angry", "surprised", "sad", "expressionless", "serious",
    "embarrassed", "pout", "worried", "scared", "sleepy", "closed_eyes", "one_eye_closed", "wink",
    "nervous", "determined", "happy", "annoyed", "shy", "confused", "excited", ":d", ":o", ";d",
]}
EXPRESSION[":d"] = "big smile"
EXPRESSION[":o"] = "open mouth"
EXPRESSION[";d"] = "winking smile"

OUTFIT = {t: t.replace("_", " ") for t in [
    "school_uniform", "serafuku", "sailor_collar", "gakuran", "blazer", "shirt", "white_shirt",
    "collared_shirt", "t-shirt", "skirt", "pleated_skirt", "miniskirt", "long_skirt", "dress",
    "sundress", "white_dress", "black_dress", "jacket", "open_jacket", "coat", "trench_coat",
    "lab_coat", "hoodie", "sweater", "cardigan", "vest", "suit", "necktie", "bowtie", "uniform",
    "military_uniform", "police_uniform", "maid", "apron", "kimono", "yukata", "japanese_clothes",
    "hakama", "chinese_clothes", "china_dress", "armor", "cape", "cloak", "robe", "shorts", "pants",
    "jeans", "overalls", "thighhighs", "pantyhose", "socks", "kneehighs", "boots", "sneakers",
    "loafers", "sandals", "gloves", "fingerless_gloves", "scarf", "hat", "witch_hat", "beret",
    "baseball_cap", "hood", "headphones", "glasses", "sunglasses", "ribbon", "hair_ribbon", "bow",
    "hair_bow", "hairband", "hair_ornament", "hairclip", "earrings", "necklace", "choker",
    "long_sleeves", "short_sleeves", "sleeveless", "detached_sleeves", "frills", "tracksuit",
    "sportswear", "gym_uniform", "jersey", "raincoat", "pajamas", "nun", "priest",
]}

# Things on the character that aren't clothing: kept as free-text "extra" tags.
EXTRA = {t: t.replace("_", " ") for t in [
    "beard", "mustache", "facial_hair", "stubble", "wrinkled_skin", "scar", "freckles", "mole",
    "animal_ears", "cat_ears", "fox_ears", "dog_ears", "rabbit_ears", "horns", "wings", "tail",
    "halo", "elf", "pointy_ears", "fang", "dark_skin", "tan", "muscular", "eyepatch", "mask",
]}

# ------------------------------------------------------------------ pose / action dynamics

# Posture comes first in a sub-action list, then limb actions, then interactions, so the
# sequence the Temporal Attention Module sees has a consistent canonical order.
POSTURE = ["standing", "sitting", "walking", "running", "jumping", "lying", "kneeling", "squatting",
           "crouching", "leaning_forward", "leaning_back", "floating", "falling", "dancing",
           "fighting_stance", "on_back", "on_stomach", "seiza", "indian_style", "flying"]
LIMB_ACTION = ["waving", "arms_up", "arm_up", "hand_up", "hands_up", "hands_on_hips", "hand_on_hip",
               "v", "double_v", "pointing", "pointing_at_viewer", "outstretched_arm", "outstretched_arms",
               "crossed_arms", "hands_in_pockets", "hand_in_pocket", "hands_together", "own_hands_together",
               "salute", "thumbs_up", "fist_pump", "clenched_hands", "arms_behind_back", "hand_on_own_chest",
               "hand_on_own_face", "hand_on_own_cheek", "head_tilt", "stretching", "reaching", "reaching_towards_viewer",
               "covering_mouth", "finger_to_mouth", "peace_sign", "heart_hands", "spread_arms", "hugging_own_legs"]
INTERACTION = ["holding", "reading", "eating", "drinking", "sleeping", "singing", "playing_instrument",
               "writing", "cooking", "fighting", "hug", "holding_hands", "carrying", "kicking", "punching",
               "swinging", "aiming", "talking", "looking_back", "looking_up", "looking_down", "looking_afar"]
ACTION_ORDER = POSTURE + LIMB_ACTION + INTERACTION
ACTION_VOCAB = {t: i for i, t in enumerate(ACTION_ORDER)}

FRAMING = {"full_body": "full_body", "cowboy_shot": "cowboy_shot", "upper_body": "upper_body",
           "portrait": "portrait", "close-up": "portrait", "feet_out_of_frame": "cowboy_shot"}
VIEW = {"from_side": "side", "from_behind": "behind", "from_above": "above", "from_below": "below"}

# ------------------------------------------------------------------ global descriptor

SCENE = {t: t.replace("_", " ") for t in [
    "outdoors", "indoors", "sky", "blue_sky", "cloud", "cloudy_sky", "night", "night_sky", "starry_sky",
    "day", "sunset", "sunrise", "evening", "twilight", "tree", "cherry_blossoms", "forest", "grass",
    "field", "flower_field", "flower", "petals", "beach", "ocean", "sea", "water", "river", "lake",
    "mountain", "snow", "snowing", "rain", "city", "cityscape", "street", "road", "building",
    "rooftop", "bridge", "park", "garden", "shrine", "torii", "temple", "castle", "ruins", "classroom",
    "school", "hallway", "library", "bookshelf", "cafe", "restaurant", "kitchen", "bedroom", "bed",
    "living_room", "couch", "chair", "desk", "window", "train", "train_station", "stairs",
    "simple_background", "white_background", "grey_background", "gradient_background",
    "black_background", "blue_background", "pink_background", "autumn_leaves", "autumn", "summer",
    "winter", "spring_(season)", "space", "underwater", "fire", "moon", "full_moon", "sun",
]}
STYLE = {t: t.replace("_", " ") for t in [
    "watercolor_(medium)", "traditional_media", "pastel_colors", "flat_color", "colorful",
    "limited_palette", "retro_artstyle", "1990s_(style)", "1980s_(style)", "2000s_(style)",
    "official_art", "anime_coloring", "backlighting", "sunlight", "light_rays", "lens_flare",
    "depth_of_field", "blurry_background", "bokeh", "dutch_angle", "chromatic_aberration",
    "dark", "silhouette", "realistic",
]}
PROP_ANIMALS = {t for t in ["cat", "dog", "bird", "rabbit", "fox", "horse", "fish", "butterfly",
                            "dragon", "black_cat", "white_cat"]}

# ------------------------------------------------------------------ selection filters

EXCLUDE = {
    # not single illustrations / no usable people
    "comic", "4koma", "2koma", "manga", "multiple_views", "reference_sheet", "character_sheet",
    "no_humans", "text_focus", "chart", "meme", "oekaki", "variations",
    # off-style or unusable media
    "monochrome", "greyscale", "sketch", "lineart", "photo_(medium)", "3d", "pixel_art", "chibi",
    "animated", "ai-generated", "ai-assisted", "cosplay_photo", "photorealistic",
    # framing that hides the body
    "head_out_of_frame", "out_of_frame", "cropped",
    # clothing levels we don't want even in general-rated images
    "swimsuit", "bikini", "underwear", "lingerie", "naked_towel", "bathing", "bath",
}
EXCLUDE_META = {"comic", "animated", "translated", "ai-generated", "ai-assisted", "photo_(medium)"}


def split_tags(s: str | None) -> list[str]:
    return s.split() if s else []


def person_counts(tags: Iterable[str]) -> tuple[int, int, bool]:
    """(girls, boys, crowd) from Danbooru count tags."""
    tags = set(tags)
    girls = max((n for t, n in GIRL_COUNT.items() if t in tags), default=0)
    boys = max((n for t, n in BOY_COUNT.items() if t in tags), default=0)
    return girls, boys, bool(tags & CROWD)


def pick(tags: Iterable[str], table: dict[str, str], limit: int | None = None) -> list[str]:
    out = [table[t] for t in tags if t in table]
    return out[:limit] if limit else out


def actions(tags: Iterable[str]) -> list[str]:
    """Ordered sub-action list (posture, limb actions, interactions). holding_* counts as holding."""
    tags = set(tags)
    if any(t.startswith("holding_") for t in tags):
        tags.add("holding")
    return [t for t in ACTION_ORDER if t in tags][:4]


def props(tags: Iterable[str]) -> list[str]:
    out = []
    for t in tags:
        if t.startswith("holding_") and t not in ("holding_hands",):
            out.append(t.removeprefix("holding_").replace("_", " "))
        elif t in PROP_ANIMALS:
            out.append(t.replace("_", " "))
    return sorted(set(out))[:6]


def framing(tags: Iterable[str]) -> str:
    for t in tags:
        if t in FRAMING:
            return FRAMING[t]
    return "unspecified"


def age(tags: Iterable[str]) -> str:
    found = [AGE_TAGS[t] for t in tags if t in AGE_TAGS]
    for a in ("elderly", "child", "adult", "teen"):
        if a in found:
            return a
    return "unspecified"


_SAFE_TAG = re.compile(r"^[a-z0-9_\-:;().'+]+$")


def is_clean(tag: str) -> bool:
    return bool(_SAFE_TAG.match(tag))
