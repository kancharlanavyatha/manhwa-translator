"""
Synthetic Manhwa Test Panel Generator.

Generates realistic high-resolution Korean webtoon panels complete with
action framing, speech balloons, Hangul dialogue, and sound effects.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 20):
    korean_font_paths = [
        "C:/Windows/Fonts/malgun.ttf",  # Malgun Gothic
        "C:/Windows/Fonts/gulim.ttc",   # Gulim
        "C:/Windows/Fonts/batang.ttc",  # Batang
    ]
    for p in korean_font_paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_manhwa_strip(output_path: Path) -> Path:
    """Create Panel 1: 800x1200 webtoon strip with 2 panels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 800, 1200
    img = Image.new("RGB", (width, height), color=(18, 18, 24))
    draw = ImageDraw.Draw(img)

    # Panel 1: Night sky
    panel1_rect = [50, 40, 750, 520]
    for y in range(panel1_rect[1], panel1_rect[3]):
        ratio = (y - panel1_rect[1]) / float(panel1_rect[3] - panel1_rect[1])
        r = int(25 + 30 * ratio)
        g = int(30 + 40 * ratio)
        b = int(60 + 60 * ratio)
        draw.line([(panel1_rect[0], y), (panel1_rect[2], y)], fill=(r, g, b))
    draw.rectangle(panel1_rect, outline=(80, 90, 120), width=3)
    draw.polygon([(100, 350), (250, 515), (70, 515)], fill=(35, 45, 75))
    draw.polygon([(600, 300), (740, 515), (550, 515)], fill=(40, 50, 85))

    # Speech Bubble 1
    bubble1_box = [120, 100, 480, 240]
    draw.ellipse(bubble1_box, fill=(255, 255, 255), outline=(20, 20, 20), width=3)
    draw.polygon([(260, 235), (285, 235), (250, 280)], fill=(255, 255, 255), outline=(20, 20, 20))
    draw.line([(261, 236), (284, 236)], fill=(255, 255, 255), width=4)

    # Panel 2: Dungeon scene
    panel2_rect = [50, 580, 750, 1140]
    for y in range(panel2_rect[1], panel2_rect[3]):
        ratio = (y - panel2_rect[1]) / float(panel2_rect[3] - panel2_rect[1])
        r = int(50 + 20 * ratio)
        g = int(20 + 15 * ratio)
        b = int(35 + 25 * ratio)
        draw.line([(panel2_rect[0], y), (panel2_rect[2], y)], fill=(r, g, b))
    draw.rectangle(panel2_rect, outline=(140, 60, 80), width=3)
    draw.ellipse([340, 850, 460, 970], fill=(160, 40, 200), outline=(230, 150, 255), width=4)

    # Speech Bubble 2
    bubble2_box = [320, 630, 700, 780]
    draw.ellipse(bubble2_box, fill=(255, 255, 255), outline=(20, 20, 20), width=3)
    draw.polygon([(460, 775), (490, 775), (440, 825)], fill=(255, 255, 255), outline=(20, 20, 20))
    draw.line([(461, 776), (489, 776)], fill=(255, 255, 255), width=4)

    font = _get_font(20)
    draw.text((160, 140), "이게... 대체 무슨 일이지?", fill=(20, 20, 24), font=font)
    draw.text((170, 175), "내가 다시 살아난 건가?!", fill=(20, 20, 24), font=font)

    draw.text((430, 675), "형, 조심해!", fill=(20, 20, 24), font=font)
    draw.text((370, 715), "뒤쪽에 괴물이 나타났어!", fill=(20, 20, 24), font=font)

    img.save(output_path, "JPEG", quality=95)
    return output_path


def create_manhwa_strip_2(output_path: Path) -> Path:
    """Create Panel 2: 800x1200 webtoon strip showing boss fight reaction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 800, 1200
    img = Image.new("RGB", (width, height), color=(18, 18, 24))
    draw = ImageDraw.Draw(img)

    # Dramatic Red/Orange Action Panel
    panel1_rect = [50, 40, 750, 560]
    for y in range(panel1_rect[1], panel1_rect[3]):
        ratio = (y - panel1_rect[1]) / float(panel1_rect[3] - panel1_rect[1])
        r = int(80 + 30 * ratio)
        g = int(25 + 20 * ratio)
        b = int(25 + 15 * ratio)
        draw.line([(panel1_rect[0], y), (panel1_rect[2], y)], fill=(r, g, b))
    draw.rectangle(panel1_rect, outline=(220, 80, 60), width=3)

    # Speech Bubble 1 in Strip 2
    bubble1_box = [160, 120, 620, 270]
    draw.ellipse(bubble1_box, fill=(255, 255, 255), outline=(20, 20, 20), width=3)
    draw.polygon([(360, 265), (390, 265), (340, 315)], fill=(255, 255, 255), outline=(20, 20, 20))
    draw.line([(361, 266), (389, 266)], fill=(255, 255, 255), width=4)

    # Bottom Panel: Awakener system window
    panel2_rect = [50, 620, 750, 1140]
    for y in range(panel2_rect[1], panel2_rect[3]):
        ratio = (y - panel2_rect[1]) / float(panel2_rect[3] - panel2_rect[1])
        r = int(20 + 20 * ratio)
        g = int(35 + 30 * ratio)
        b = int(60 + 50 * ratio)
        draw.line([(panel2_rect[0], y), (panel2_rect[2], y)], fill=(r, g, b))
    draw.rectangle(panel2_rect, outline=(70, 130, 220), width=3)

    # System UI Box (Narration style)
    system_box = [150, 780, 650, 920]
    draw.rectangle(system_box, fill=(255, 255, 255), outline=(30, 60, 140), width=3)

    font = _get_font(20)
    draw.text((210, 160), "말도 안 돼... 저 녀석의", fill=(20, 20, 24), font=font)
    draw.text((220, 195), "마력이 측정 불가라니?!", fill=(20, 20, 24), font=font)

    font_sys = _get_font(18)
    draw.text((180, 830), "시스템: 각성자의 능력이 활성화되었습니다.", fill=(15, 25, 60), font=font_sys)

    img.save(output_path, "JPEG", quality=95)
    return output_path


if __name__ == "__main__":
    out1 = Path("tests/fixtures/sample_panel_1.jpg")
    out2 = Path("tests/fixtures/sample_panel_2.jpg")
    create_manhwa_strip(out1)
    create_manhwa_strip_2(out2)

    # Also copy to demo/assets
    demo_dir = Path("demo/assets")
    demo_dir.mkdir(parents=True, exist_ok=True)
    create_manhwa_strip(demo_dir / "panel_1.jpg")
    create_manhwa_strip_2(demo_dir / "panel_2.jpg")
    print(f"Generated sample panels in tests/fixtures and demo/assets")
