from PIL import Image, ImageDraw
import math

S = 360
img = Image.new("RGB", (S, S), (245, 245, 245))
d = ImageDraw.Draw(img)

def sc(x, y):
    return (x * 1.5, y * 1.5)

# 1. ellipse - purple
d.ellipse([sc(30, 30), sc(150, 130)], fill=(150, 40, 170))

# 2. rectangle - teal
d.rectangle([sc(160, 40), sc(220, 110)], fill=(20, 170, 160))

# 3. triangle - orange
d.polygon([sc(60, 200), sc(120, 130), sc(180, 200)], fill=(230, 130, 30))

# 4. star - blue
cx, cy, R, r = 190, 180, 45, 18
pts = []
for i in range(10):
    ang = -math.pi/2 + i * math.pi/5
    rad = R if i % 2 == 0 else r
    pts.append((cx + rad*math.cos(ang), cy + rad*math.sin(ang)))
d.polygon(pts, fill=(40, 80, 220))

# 5. small circle - red, overlaps the triangle a bit
d.ellipse([sc(150, 150), sc(200, 200)], fill=(210, 40, 40))

img.save("test_sample.png")
print("saved test_sample.png", img.size)
