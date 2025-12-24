import PyInstaller.__main__
import os
import shutil

def build():
    print("Building Executable...")
    
    # Clean previous builds
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")

    # PyInstaller arguments
    args = [
        'main.py',                      # Script to bundle
        '--name=EagleEye',              # Exe name
        '--onedir',                     # Directory bundler (faster startup, easier config)
        '--noconfirm',                  # Overwrite
        '--clean',                      # Clean cache
        # Data files: Source -> Dest
        '--add-data=dashboard/templates;dashboard/templates', 
        '--add-data=models;models',
        '--add-data=settings.yaml;.', 
        '--add-data=.env.example;.',
        '--add-data=milvus_demo_local.json;.',
        # Hidden imports (often missed by auto-analysis)
        '--hidden-import=engineio.async_drivers.threading',
        '--hidden-import=flask_socketio',
        '--hidden-import=pynmea2',
        '--hidden-import=serial',
        '--hidden-import=winsdk',
        '--hidden-import=onnxruntime_extensions',
        '--hidden-import=insightface',
        '--hidden-import=sklearn.utils._cython_blas',
        '--hidden-import=sklearn.neighbors.typedefs',
        '--hidden-import=sklearn.neighbors.quad_tree',
        '--hidden-import=sklearn.tree',
        '--hidden-import=sklearn.tree._utils',
    ]
    
    PyInstaller.__main__.run(args)
    
    print("\nCheck 'dist/EagleEye' folder.")

if __name__ == "__main__":
    build()
