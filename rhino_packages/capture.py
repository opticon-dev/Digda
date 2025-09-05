import Rhino
import Rhino.Geometry as geo
import System.Drawing as drawing


def capture_render_view(rhino_doc, width=1920, height=1080, dpi=96):
    # 현재 뷰 가져오기
    view = rhino_doc.Views.ActiveView
    vp = view.ActiveViewport

    # Rendered DisplayMode 찾기
    display_mode = Rhino.Display.DisplayModeDescription.FindByName("Rendered")
    if display_mode is None:
        raise Exception("Rendered display mode not found.")

    # 뷰포트 DisplayMode를 Rendered로 변경
    vp.DisplayMode = display_mode
    # 다시 그리기
    view.Redraw()

    # ViewCapture 설정
    capture = Rhino.Display.ViewCapture()
    capture.Width = width
    capture.Height = height
    capture.ScaleScreenItems = False
    capture.DrawAxes = False
    capture.DrawGrid = False
    capture.DrawGridAxes = False
    capture.RealtimeRenderPasses = 1
    capture.RealtimeRenderFrameRate = 30
    capture.RealtimeRenderCycles = 1

    # 캡처 실행
    bmp = capture.CaptureToBitmap(view)
    if bmp is None:
        raise Exception("View capture failed.")
    return bmp
