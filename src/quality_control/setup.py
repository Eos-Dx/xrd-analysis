from setuptools import setup, find_packages

setup(
    name="EosDXQualityTool",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "joblib",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "eosdx-quality-tool = eosdx_quality_tool.main:main"
        ]
    },
)
