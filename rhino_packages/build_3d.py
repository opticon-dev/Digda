import Rhino.Geometry as geo
from .constants import *
from .utils import set_clockwise, offset_closed_crv_inside, offset_closed_crv_outside, extrude_crv, brep_boolean_difference
from typing import List, Optional, Tuple

def make_door_from_open_curve(d_crv, height, thickness):
    if not d_crv:
        return None

    offset1 = d_crv.Offset(geo.Plane.WorldXY,  thickness/2, 0.01, geo.CurveOffsetCornerStyle.Sharp)
    offset2 = d_crv.Offset(geo.Plane.WorldXY, -thickness/2, 0.01, geo.CurveOffsetCornerStyle.Sharp)
    if not offset1 or not offset2:
        return None

    c1, c2 = offset1[0], offset2[0]

    # 양 끝 연결
    side1 = geo.Line(c1.PointAtStart, c2.PointAtStart).ToNurbsCurve()
    side2 = geo.Line(c1.PointAtEnd,   c2.PointAtEnd).ToNurbsCurve()

    # 닫힌 경계
    curves = [c1, side2, c2, side1]
    joined = geo.Curve.JoinCurves(curves)
    if not joined or len(joined) == 0:
        return None

    profile = joined[0]
    if not profile.IsClosed:
        return None

    # Extrude (높이)
    extrusion = geo.Extrusion.Create(profile, height, True)
    if not extrusion:
        raise ValueError("not door extrusion")
    return extrusion.ToBrep()

def make_window_from_open_curve(w_crv, thickness, height, height_from_bottom):
    if not w_crv:
        return None

    # 1. Offset (양쪽)
    offset1 = w_crv.Offset(geo.Plane.WorldXY,  thickness/2, 0.01, geo.CurveOffsetCornerStyle.Sharp)
    offset2 = w_crv.Offset(geo.Plane.WorldXY, -thickness/2, 0.01, geo.CurveOffsetCornerStyle.Sharp)
    if not offset1 or not offset2:
        return None

    c1, c2 = offset1[0], offset2[0]

    # 2. 양 끝점 잇기
    side1 = geo.Line(c1.PointAtStart, c2.PointAtStart).ToNurbsCurve()
    side2 = geo.Line(c1.PointAtEnd,   c2.PointAtEnd).ToNurbsCurve()

    # 3. 닫힌 경계 만들기
    curves = [c1, side2, c2, side1]
    joined = geo.Curve.JoinCurves(curves)
    if not joined or len(joined) == 0:
        return None

    profile = joined[0]
    if not profile.IsClosed:
        return None

    # 4. 창 하단 높이만큼 Z 방향으로 이동
    move = geo.Transform.Translation(0, 0, height_from_bottom)
    profile_dup = profile.DuplicateCurve()
    profile_dup.Transform(move)

    # 5. Extrude (창 높이)
    extrusion = geo.Extrusion.Create(profile_dup, height, True)
    if not extrusion:
        raise ValueError("not window extrusion")
    return extrusion.ToBrep()

def get_room_wall(segs, thickness, height):
    # type: (list[geo.Curve], float, float) -> geo.Brep
    """
    벽들의 단일 세그먼트로부터, 두께와 높이만큼 벽을 만든다.
    """
    segs = set_clockwise(segs)
    # TODO: segs에 순서대로 내부 세그, 외부 세그 나오도록 설정
    walls = [] # type: list[geo.Brep]

    for seg in segs:
        inner_offset = offset_closed_crv_inside(seg, thickness/2)
        outer_offset = offset_closed_crv_outside(seg, thickness/2)

        inner_extrude = extrude_crv(inner_offset, height)
        outer_extrude = extrude_crv(outer_offset, height)

        wall = brep_boolean_difference(outer_extrude, inner_extrude)
        walls.append(wall)
    wall_unions = geo.Brep.CreateBooleanUnion(walls, 0.01)

    if len(wall_unions) > 1:
        raise ValueError("벽이 연결되어 있지 않습니다.")
    wall_union = wall_unions[0]
    return wall_union
        
def get_doors(segs, thickness, height):
    # type: (list[geo.Curve], float, float) -> list[geo.Brep]
    """
    문들의 단일 세그먼트로부터, 두께와 높이만큼 문을 만든다.
    """
    doors = [] # type: list[geo.Brep]
    for seg in segs:
        door = make_door_from_open_curve(seg, thickness, height)
        doors.append(door)

def get_windows(segs, thickness, height, height_from_bottom):
    # type: (list[geo.Curve], float, float, float) -> list[geo.Brep]
    """
    창문들의 세그먼트로부터, 바닥의 높이에서 창문의 높이, 두께만큼 만든다.
    """
    windows = [] # type: list[geo.Brep]
    for seg in segs:
        window = make_window_from_open_curve(seg, thickness, height, height_from_bottom)
        windows.append(window)
    return windows

def get_bottom_slab(wall_region, thickness):
    # type: (geo.Curve, float) -> geo.Brep
    """
    천장 슬래브
    """
    outside_offset_crv = offset_closed_crv_outside(wall_region, thickness/2)
    extrusion = geo.Extrusion.Create(outside_offset_crv, -thickness, True)
    if not extrusion:
        raise ValueError("not bottom extrusion")
    return extrusion.ToBrep()

def get_top_slab(wall_region, thickness, wall_height):
    # type: (geo.Curve, float, float) -> geo.Brep
    """
    바닥 슬래브
    """
    outside_offset_crv = offset_closed_crv_outside(wall_region, thickness/2)
    base = outside_offset_crv.DuplicateCurve()
    move = geo.Transform.Translation(0, 0, wall_height)  # 벽 높이만큼 위로 이동
    base.Transform(move)

    extrusion = geo.Extrusion.Create(base, thickness, True)
    if not extrusion:
        raise ValueError("not top extrusion")
    return extrusion.ToBrep()

def build(wall_regions, door_segs, window_segs):
    # type: (list[geo.Curve], list[geo.Curve], list[geo.Curve]) -> tuple[list[geo.Brep], list[geo.Brep], list[geo.Brep]]
    """
    문과 창문을 제외한 벽체를 생성한다.
    """

    wall = get_room_wall(wall_regions, wall_thickness, wall_height)
    doors = get_doors(door_segs, 300, 3000)
    windows = get_windows(window_segs, wall_thickness, window_height, window_height_from_bottom)

    print(doors)
    print(windows)

    if not doors:
        doors = []
    if not windows:
        windows = []

    cutters = doors + windows
    refined_wall = geo.Brep.CreateBooleanDifference([wall], cutters, TOL)

    bottom_slabs = [get_bottom_slab(wr, slab_thickness) for wr in wall_regions]
    top_slabs = [get_top_slab(wr, slab_thickness, wall_height) for wr in wall_regions]

    bottom_slab = geo.Brep.CreateBooleanUnion(bottom_slabs, 0.01)
    top_slab = geo.Brep.CreateBooleanUnion(top_slabs, 0.01)

    return refined_wall, bottom_slab, top_slab