@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem  ComfyUI Gemini/External API 节点自动安装脚本
rem  作者：zhanglongxiao111（基于 Aryan185 的原始项目）
rem  用途：在固定目录下克隆/更新节点仓库，并安装所需依赖
rem ============================================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "COMFY_ROOT=%SCRIPT_DIR%"
set "PYTHON_EXE=%COMFY_ROOT%\python\python.exe"
set "CUSTOM_NODES_DIR=%COMFY_ROOT%\custom_nodes"
set "REPO_DIR=%CUSTOM_NODES_DIR%\Comfyui-geminiapi"
set "REPO_URL=https://github.com/zhanglongxiao111/Comfyui-geminiapi.git"

echo.
echo [1/5] 检查 Python 解释器路径...
if not exist "%PYTHON_EXE%" (
    echo  错误：未找到 %PYTHON_EXE%
    echo  请确认 ComfyUI 安装路径是否正确。
    exit /b 1
)
for /f "delims=" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do (
    echo  使用 Python: %%v
)

echo.
echo [2/5] 确认 custom_nodes 目录...
if not exist "%CUSTOM_NODES_DIR%" (
    echo  未找到 custom_nodes 目录，正在创建...
    mkdir "%CUSTOM_NODES_DIR%"
)

echo.
echo [3/5] 获取或更新仓库...
if exist "%REPO_DIR%\.git" (
    pushd "%REPO_DIR%" >nul
    echo  仓库已存在，执行 git pull 更新...
    git pull
    if errorlevel 1 (
        echo  git pull 失败，请手动检查后重试。
        popd >nul
        exit /b 1
    )
    popd >nul
) else (
    pushd "%CUSTOM_NODES_DIR%" >nul
    echo  克隆仓库：%REPO_URL%
    git clone "%REPO_URL%"
    if errorlevel 1 (
        echo  git clone 失败，请检查网络或 Git 配置。
        popd >nul
        exit /b 1
    )
    popd >nul
)

echo.
echo [4/5] 安装 Python 依赖...
if not exist "%REPO_DIR%\requirements.txt" (
    echo  未找到 requirements.txt ，跳过依赖安装。
) else (
    pushd "%REPO_DIR%" >nul
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  依赖安装过程中出现错误，请检查日志。
        popd >nul
        exit /b 1
    )
    popd >nul
)

echo.
echo [5/5] 完成。请重启 ComfyUI 使节点生效。
echo.
exit /b 0
