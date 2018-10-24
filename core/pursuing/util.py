"""
Created on 2016-11-29

@author: Peer Springstübe
"""


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


def loc2xgeo(location):
    x, y, w, h = location
    cx = x + (w - 1) / 2
    cy = y + (h - 1) / 2
    gw = w / 1
    gh = h / 1
    geo = [cx, cy, gw, gh]
    return geo


def xgeo2loc(geo):
    cx, cy, pw, ph = geo
    w = pw * 1
    h = ph * 1
    x = cx - (w - 1) / 2
    y = cy - (h - 1) / 2
    return [x, y, w, h]