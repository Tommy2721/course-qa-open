@echo off
cd /d "%~dp0"

echo ============================================
echo   校准拒答阈值
echo ============================================
echo.
echo 为什么必须做这一步：
echo   拒答阈值决定「什么问题该答、什么问题该拒」。
echo   换个课件，分数分布就变了，阈值必须跟着变。
echo.
echo 需要先准备好评测集 eval\questions.jsonl
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 没找到虚拟环境 .venv
    pause
    exit /b 1
)

if not exist "eval\questions.jsonl" (
    echo [错误] 没找到 eval\questions.jsonl
    pause
    exit /b 1
)

echo 正在扫描阈值，约需几分钟...
echo.
.venv\Scripts\python.exe -m src.eval --calibrate

echo.
echo --------------------------------------------
echo 往上翻，找到「推荐阈值」那行和分数分布表（截图保存），
echo 把推荐数字填到 config.py 的 MIN_RERANK_SCORE = 那里，
echo 保存后重新双击「启动界面.bat」生效。
echo --------------------------------------------
echo.
pause
