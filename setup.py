import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="guardbench",
    version="1.0.0",
    author="Elias Bassani",
    author_email="elias.bassani@ec.europa.eu",
    description="GuardBench: A Large-Scale Benchmark for Guardrail Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
        "datasets",
        "huggingface_hub",
        "imbalanced-learn",
        "ipython",
        "loguru",
        "scikit-learn",
        "tabulate",
        "tqdm",
        "unified_io",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: General",
    ],
    keywords=[
        "harmful content detection",
        "harmful content",
        "toxic content detection",
        "toxic content",
        "guardrails",
        "guardrail models",
        "guardrail models evaluation",
        "guardrail models benchmark",
        "evaluation",
        "benchmark",
    ],
    python_requires=">=3.10",
)
