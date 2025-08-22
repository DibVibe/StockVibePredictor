from setuptools import setup, find_packages

setup(
    name="stockvibepredictor",
    version="1.0.0",
    author="Dibakar",
    author_email="dibvibe.dev@gmail.com",
    description="AI-powered stock market prediction system using Django and ML",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ThisIsDibakar/StockVibePredictor",
    project_urls={
        "Bug Tracker": "https://github.com/DibVibe/StockVibePredictor/issues",
        "Documentation": "https://github.com/DibVibe/StockVibePredictor/wiki",
        "Source Code": "https://github.com/DibVibe/StockVibePredictor",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Framework :: Django :: 5.0",
    ],
    python_requires=">=3.8",
    install_requires=[
        "django>=5.0",
        "djangorestframework>=3.14",
        "yfinance>=0.2.18",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "redis>=4.5.0",
    ],
    maintainer="Dibakar",
    keywords="stock-prediction machine-learning django react financial-analysis",
)
