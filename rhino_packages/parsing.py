import Rhino.Geometry as geo
from typing import Union, Optional, List, Dict, Any


class JsonToGeometry:
    """
    JSON 구조(rooms, doors, windows)를 받아
    Rhino.Geometry Curve 객체로 변환하는 클래스
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.rooms: Dict[str, geo.Curve] = {}
        self.doors: List[geo.Curve] = []
        self.windows: List[geo.Curve] = []
        self._parse_all()

    # --- 내부 유틸 ---
    def _parse_coords(self, coord_str: str) -> List[geo.Point3d]:
        pts: List[geo.Point3d] = []
        for c in coord_str.split(","):
            nums = list(map(float, c.strip().split()))
            if len(nums) == 2:  # x, y만 있는 경우 → z=0
                x, y = nums
                pts.append(geo.Point3d(x, y, 0.0))
            elif len(nums) == 3:
                x, y, z = nums
                pts.append(geo.Point3d(x, y, z))
            else:
                raise ValueError(f"좌표 포맷 오류: {nums}")
        return pts

    def parse_polygon(self, wkt: str) -> geo.Curve:
        coords = wkt.replace("POLYGON ((", "").replace("))", "")
        pts = self._parse_coords(coords)
        return geo.Polyline(pts).ToPolylineCurve()

    def parse_linestring(self, wkt: str) -> geo.Curve:
        coords = wkt.replace("LINESTRING (", "").replace(")", "")
        pts = self._parse_coords(coords)
        if len(pts) == 2:
            return geo.Line(pts[0], pts[1]).ToNurbsCurve()
        return geo.Polyline(pts).ToPolylineCurve()

    def parse_point(self, wkt: str) -> geo.Point3d:
        coord = wkt.replace("POINT (", "").replace(")", "")
        nums = list(map(float, coord.strip().split()))
        if len(nums) == 2:
            x, y = nums
            return geo.Point3d(x, y, 0.0)
        elif len(nums) == 3:
            x, y, z = nums
            return geo.Point3d(x, y, z)
        else:
            raise ValueError(f"좌표 포맷 오류: {nums}")

    def parse(self, wkt: str) -> Optional[Union[geo.Curve, geo.Point3d]]:
        if wkt.startswith("POLYGON"):
            return self.parse_polygon(wkt)
        elif wkt.startswith("LINESTRING"):
            return self.parse_linestring(wkt)
        elif wkt.startswith("POINT"):
            return self.parse_point(wkt)
        return None

    # --- 메인 파서 ---
    def _parse_all(self) -> None:
        """JSON 구조 전체 파싱"""
        # rooms
        for r in self.data.get("rooms", []):
            geom = self.parse(r["geom"])
            if isinstance(geom, geo.Curve):
                self.rooms[r["room_name"]] = geom

        # doors
        for d in self.data.get("doors", []):
            geom = self.parse(d["geom"])
            if isinstance(geom, geo.Curve):
                self.doors.append(geom)

        # windows
        for w in self.data.get("windows", []):
            geom = self.parse(w["geom"])
            if isinstance(geom, geo.Curve):
                self.windows.append(geom)

