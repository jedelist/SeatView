def bb_center(x1,y1,x2,y2): return ((x1+x2)//2, (y1+y2)//2)

def inflate_box(x1,y1,x2,y2, pct=0.15):
    w, h = x2-x1, y2-y1
    dx, dy = int(pct*w), int(pct*h)
    return x1-dx, y1-dy, x2+dx, y2+dy

# Collect boxes by class
tables, chairs, people, bags = [], [], [], []
for b in r.boxes:
    cls = int(b.cls[0]); x1,y1,x2,y2 = map(int, b.xyxy[0])
    if cls == 60: tables.append((x1,y1,x2,y2))
    elif cls == 56: chairs.append((x1,y1,x2,y2))
    elif cls == 0:  people.append((x1,y1,x2,y2))
    elif cls in (24,26,28): bags.append((x1,y1,x2,y2))

# Mark a table occupied if a chair center lies inside its inflated box,
# AND (optionally) there is a person or bag near that chair.
import math

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1,ix2,iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih; areaA=(ax2-ax1)*(ay2-ay1); areaB=(bx2-bx1)*(by2-by1)
    return inter / max(1, (areaA+areaB-inter))

occupied = [False]*len(tables)
for ti, tb in enumerate(tables):
    tb_inf = inflate_box(*tb, pct=0.20)
    # find chairs that sit within table footprint
    chairs_near = []
    for cb in chairs:
        cx, cy = bb_center(*cb)
        x1,y1,x2,y2 = tb_inf
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            chairs_near.append(cb)
    # check for people/bags overlapping those chairs
    for cb in chairs_near:
        # person or bag overlapping chair box implies "occupied seat"
        if any(iou(cb, pb) > 0.15 for pb in people) or any(iou(cb, bb) > 0.15 for bb in bags):
            occupied[ti] = True
            break
