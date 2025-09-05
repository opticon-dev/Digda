import Rhino.Geometry as geo

def set_clockwise(crvs):
    update_crvs = [] # type: list[geo.Curve]
    for crv in crvs:
        orientation = crv.ClosedCurveOrientation(geo.Plane.WorldXY)
        if orientation == geo.CurveOrientation.CounterClockwise:
            update_crvs.append(crv)
        else:
            crv.Reverse()
            update_crvs.append(crv)
    return update_crvs

def is_crv_clockwise(crv):
    orientation = crv.ClosedCurveOrientation(geo.Plane.WorldXY)
    if orientation == geo.CurveOrientation.CounterClockwise:
        return True
    elif orientation == geo.CurveOrientation.Clockwise:
        return False
    else:
        return None

def _try_get_plane(crv):
    plane = geo.Plane.Unset

    ok, plane = crv.TryGetPlane()
    return (ok, plane)

def offset_closed_crv_inside(crv, dist):
    ok, plane = _try_get_plane(crv)
    if not ok or not crv.IsClosed:
        return None
    if is_crv_clockwise(crv):
        dist = -dist
    segs = crv.Offset(plane, dist, 0.01, geo.CurveOffsetCornerStyle.Sharp)
    if not segs: return None
    joined = geo.Curve.JoinCurves(segs, 0.01)
    if len(joined) != 1: return None
    return joined[0]

def offset_closed_crv_outside(crv, dist):

    ok, plane = _try_get_plane(crv)
    if not ok or not crv.IsClosed:
        return None
    if not is_crv_clockwise(crv):
        dist = -dist
    segs = crv.Offset(plane, dist, 0.01, geo.CurveOffsetCornerStyle.Sharp)
    if not segs: return None
    joined = geo.Curve.JoinCurves(segs, 0.01)
    if len(joined) != 1: return None
    return joined[0]

def extrude_crv(crv, height):
    if not crv: return None
    ext = geo.Extrusion.Create(crv, height, True)
    return ext.ToBrep() if ext else None

def brep_boolean_difference(base_brep, cutter_brep):
    if not base_brep or not cutter_brep:
        return None
    if isinstance(base_brep, list) and len(base_brep) > 0:
        base_brep = base_brep[0]
    if isinstance(cutter_brep, list) and len(cutter_brep) > 0:
        cutter_brep = cutter_brep[0]
    try:
        tol = Rhino.RhinoDoc.ActiveDoc.ModelAbsoluteTolerance
    except:
        tol = 1e-3
    result = geo.Brep.CreateBooleanDifference(base_brep, cutter_brep, tol)
    if result and len(result) > 0:
        return result[0]
    return None