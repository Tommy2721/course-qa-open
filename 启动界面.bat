@echo off
cd /d "%~dp0"

echo ============================================
echo   课问 - 课程知识问答与复习助手
echo   AIC 2026 算法主题赛（AI+开源）
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 没找到虚拟环境 .venv
    pause
    exit /b 1
)

echo [1/2] 检查本地大模型服务...
curl -s --max-time 4 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo       Ollama 没在运行，正在启动...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
        timeout /t 6 /nobreak >nul
    ) else (
        echo       [警告] 没找到 Ollama，回答功能会不可用。
        echo       请先安装：https://ollama.com/download/windows
    )
) else (
    echo       Ollama 正常
)

echo.
echo [2/2] 启动界面...
echo.
echo   浏览器会自动打开；若没有，手动访问 http://localhost:8501
echo   首次打开需要十几秒加载模型，之后就快了。
echo.
echo   关闭服务：在本窗口按 Ctrl+C
echo   提示：演示前先随便问一个问题预热。
echo.

.venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none --client.toolbarMode minimal --browser.gatherUsageStats false

pause
