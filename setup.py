"""
Setup script for AI Trading System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-trading-system",
    version="1.0.0",
    author="AI Trading Team",
    author_email="team@aitrading.com",
    description="Advanced AI-powered automated trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "yfinance>=0.1.70",
        "torch>=1.9.0",
        "plotly>=5.5.0",
        "flask>=2.0.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-trading=cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.html", "*.css", "*.js"],
    },
)