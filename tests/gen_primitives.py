from PIL import Image, ImageDraw
import math, json, colorsys

W=800
img=Image.new('RGB',(W,W),'black')
dr=ImageDraw.Draw(img)

def polygon(cx,cy,r,n,rot=0):
    return [(cx+r*math.cos(rot+2*math.pi*i/n),cy-r*math.sin(rot+2*math.pi*i/n)) for i in range(n)]

def star(cx,cy,r,n):
    pts=[]
    for i in range(2*n):
        a=math.pi/2+math.pi*i/n
        rad=r if i%2==0 else r*0.4
        pts.append((cx+rad*math.cos(a),cy-rad*math.sin(a)))
    return pts

def spline_pts(cx,cy,w,h,n=60):
    return [(cx+w*h*math.cos(t)/math.hypot(h*math.cos(t),w*math.sin(t)),cy-w*h*math.sin(t)/math.hypot(h*math.cos(t),w*math.sin(t))) for t in [2*math.pi*i/n for i in range(n)]]

def blob(cx,cy,r,n=60,amp=0.15,freq=4):
    return [(cx+r*(1+amp*math.sin(freq*t+0.5*math.sin(2*t)))*math.cos(t),cy-r*(1+amp*math.sin(freq*t+0.5*math.sin(2*t)))*math.sin(t)) for t in [2*math.pi*i/n for i in range(n)]]

def lens_lune_pts(cx,cy,r,offset,n=60,lens=True):
    d2=offset/2.0; h=math.sqrt(max(0,r*r-d2*d2))
    cx2=cx+offset
    a_up=math.atan2(-h,d2); a_dn=math.atan2(h,d2)
    b_dn=math.atan2(h,-d2); b_up=math.atan2(-h,-d2)
    if b_up<0: b_up+=2*math.pi
    half=n//2
    pts=[]
    if lens:
        for i in range(half+1):
            t=a_up+(a_dn-a_up)*i/half
            pts.append((cx+r*math.cos(t),cy+r*math.sin(t)))
    else:
        for i in range(half+1):
            t=a_up+(a_dn-2*math.pi-a_up)*i/half
            pts.append((cx+r*math.cos(t),cy+r*math.sin(t)))
    for i in range(1,half+1):
        t=b_dn+(b_up-b_dn)*i/half
        pts.append((cx2+r*math.cos(t),cy+r*math.sin(t)))
    return pts

layout=[
    ('circle',    100,130,60,None),
    ('triangle',  300,130,65,None),
    ('rect',      500,130,55,'square'),
    ('pentagon',  700,130,60,None),
    ('hexagon',   100,400,60,None),
    ('star',      300,400,60,5),
    ('irregular', 500,400,0,None),
    ('concave',   700,400,0,None),
    ('blob',      100,670,55,None),
    ('ellipse',   300,670,0,(50,30)),
    ('lune',      500,670,55,35),
    ('lens',      700,670,55,35),
]

# Evenly HSV-spaced, S=255, V=200
colors=[tuple(round(255*c) for c in colorsys.hsv_to_rgb(i/12,1.0,200/255)) for i in range(12)]

shapes=[]
for (typ,cx,cy,r,ex),col in zip(layout,colors):
    if typ=='circle': pts=polygon(cx,cy,r,60)
    elif typ=='triangle': pts=polygon(cx,cy,r,3)
    elif typ=='rect': pts=polygon(cx,cy,r,4,rot=math.pi/4)
    elif typ=='pentagon': pts=polygon(cx,cy,r,5)
    elif typ=='hexagon': pts=polygon(cx,cy,r,6)
    elif typ=='star': pts=star(cx,cy,r,ex)
    elif typ=='irregular': pts=[(430,320),(490,360),(550,450),(520,530),(400,490)]
    elif typ=='concave': pts=[(620,330),(680,360),(710,430),(760,480),(700,530),(650,420)]
    elif typ=='blob': pts=blob(cx,cy,r)
    elif typ=='ellipse': pts=spline_pts(cx,cy,*ex)
    elif typ=='lune': pts=lens_lune_pts(cx,cy,r,ex,60,lens=False)
    elif typ=='lens': pts=lens_lune_pts(cx,cy,r,ex,60,lens=True)
    shapes.append({'type':typ,'pts':[[round(p[0],1),round(p[1],1)] for p in pts],'color':list(col)})
    dr.polygon(pts,fill=col)

img.save('tests/primitives.png')
with open('tests/primitives.json','w') as f:
    json.dump(shapes,f)
print('done')
