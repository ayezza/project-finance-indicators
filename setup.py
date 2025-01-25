from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define requirements directly
requirements = [
    "matplotlib>=3.10.0",
    "numpy>=2.2.2",
    "numpy-financial>=1.0.0",
    "pandas>=2.2.3",
    "scikit-learn>=1.6.1",
    "scipy>=1.15.1",
    "seaborn>=0.13.2",
    "sympy>=1.13.3"
]

setup(
    name="project-finance-indicators",
    version="0.1.0",
    author="ayezza",
    description="A Python package for financial calculations and indicators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayezza/project-finance-indicators",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
