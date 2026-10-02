"""Web 层的后端。

与 :mod:`forge_core` 的边界
--------------------------
* ``forge_core`` 是纯 ML 库，**绝不** import fastapi / uvicorn；
* ``forge_web`` 只做三件事：解析请求、调用 ``forge_core``、把结果转成 JSON。

因此 Web 层可以被整体删掉而不影响训练与 CLI，反之亦然。
"""

__all__ = ["app", "main"]
