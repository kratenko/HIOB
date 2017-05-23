def loc2affgeo(location, particle_size):
    x, y, w, h = location
    cx = x + (w - 1) / 2
    cy = y + (h - 1) / 2
    gw = w / particle_size
    gh = h / w
    geo = [cx, cy, gw, gh]
    return geo


def affgeo2loc(geo, particle_size):
    cx, cy, pw, ph = geo
    w = pw * particle_size
    h = ph * w
    x = cx - (w - 1) / 2
    y = cy - (h - 1) / 2
    return [x, y, w, h]

particle_size = 64
loc = [400, 48, 87, 319]
geo1 = loc2affgeo(loc, particle_size)
print(loc, geo1)

geo = [4.423438512426096e+02, 2.044431913463012e+02,
       1.359375000000000, 3.666666666666667]
loc = affgeo2loc(geo, particle_size)
print(geo, loc)
