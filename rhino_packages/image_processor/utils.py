import Rhino


import io
import System
from System.Drawing import Imaging, Bitmap
from System.IO import MemoryStream
from System.Drawing import Image as SystemImage
from System import Array, Byte


def bitmap_to_bytesio(bmp, format=Imaging.ImageFormat.Png):
    from PIL import Image

    bmp.Save("tmp.png", format)
    img = Image.open("tmp.png").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    # 파일명 지정(업로더가 MIME 추정하기 쉬움)
    buf.name = "input.jpg"
    return buf


def python_byte_to_Dotnet_bitmap(python_byte):

    # Python bytes → .NET byte[] 변환
    net_bytes = Array[Byte](python_byte)

    # 이제 MemoryStream 생성
    ms = MemoryStream(net_bytes)

    # 이미지 로드
    img = SystemImage.FromStream(ms, True, True)
    bmp = Bitmap(img)  # 독립된 Bitmap 복제

    # 리소스 정리
    img.Dispose()
    ms.Close()

    return bmp
