import numpy as np

particle_count = 6

a = np.array([10, 10, 0, 0, 0, 0])

aa = np.tile(a, (particle_count, 1)).T

print(aa)

r = np.random.randn(6, particle_count)

print(r)

rn = np.multiply(r, aa)

print(rn)


def loc2affgeo(location, particle_size=64):
    x, y, w, h = location
    cx = x + (w - 1) / 2
    cy = y + (h - 1) / 2
    gw = w / particle_size
    gh = h / w
    geo = [cx, cy, gw, gh]
    return geo


def affgeo2loc(geo, particle_size=64):
    cx, cy, pw, ph = geo
    w = pw * particle_size
    h = ph * w
    x = cx - (w - 1) / 2
    y = cy - (h - 1) / 2
    return [x, y, w, h]

l = (116, 68, 122, 125)
g = loc2affgeo(l)
l2 = affgeo2loc(g)

print(l, g, l2)


def generate_geo_particles(geo):
    # geo = loc2affgeo(loc)
    particle_count = 10
    geos = np.tile(geo, (particle_count, 1)).T
    r = np.random.randn(4, particle_count)
    f = np.tile([10, 10, 0, 0], (particle_count, 1)).T
    rn = np.multiply(r, f)
    return geos + rn

geo = (140, 35, 1.9, 1.5)
b = generate_geo_particles(geo)
print(np.shape(b))
print(b.T)