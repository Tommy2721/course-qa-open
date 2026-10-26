@echo off
cd /d "%~dp0"

echo ============================================
echo   重建课程知识库索引
echo ============================================
echo.
echo 这一步会：解析 data\raw\ 下的所有课件，分块，向量化，建关键词索引
echo.
echo 提示：平时更方便的做法是在网页界面的「课件管理」页签里
echo       上传文件并点「保存并重建索引」，效果一样，还不用敲命令。
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 没找到虚拟环境 .venv
    pause
    exit /b 1
)

if not exist "data\raw" (
    echo [错误] 没找到 data\raw 目录，请把课件放进去。
    pause
    exit /b 1
)

echo 正在处理，课件多的话需要几分钟，请勿关闭窗口...
echo.
.venv\Scripts\python.exe -m src.index

if errorlevel 1 (
    echo.
    echo [失败] 请把上面的报错信息保存下来排查。
    pause
    exit /b 1
)

echo.
echo 完成。接下来建议：
echo   1. 双击「校准拒答阈值.bat」重新校准
echo   2. 双击「启动界面.bat」查看效果
echo.
pause
