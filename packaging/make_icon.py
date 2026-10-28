"""Draws packaging/icon.png (1024 px): a dust disk with a bright ring around a star.

Run once when the icon changes: python packaging/make_icon.py  (needs Pillow)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((40, 40, S - 40, S - 40), radius=210, fill=(20, 33, 58, 255))

disk = Image.new("RGBA", (S, S), (0, 0, 0, 0))
dd = ImageDraw.Draw(disk)
cx, cy = S / 2, S / 2
for rx, w, a in ((420, 6, 90), (360, 5, 120), (300, 4, 140), (175, 4, 150)):
    dd.ellipse((cx - rx, cy - rx * 0.30, cx + rx, cy + rx * 0.30), outline=(240, 200, 150, a), width=w)
dd.ellipse((cx - 240, cy - 72, cx + 240, cy + 72), outline=(217, 130, 43, 255), width=26)  # trap ring
disk = disk.filter(ImageFilter.GaussianBlur(0.8))
img.alpha_composite(disk)

glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(glow).ellipse((cx - 120, cy - 120, cx + 120, cy + 120), fill=(242, 180, 90, 150))
img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(40)))
ImageDraw.Draw(img).ellipse((cx - 62, cy - 62, cx + 62, cy + 62), fill=(255, 236, 196, 255))

out = Path(__file__).with_name("icon.png")
img.save(out)
print("wrote", out)
