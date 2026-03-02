from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="transcribe",
    version="0.1.0",
    description="A Tri-Agent Ontology Framework for Automated Annotation of Transcriptomics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "transcribe=transcribe.cli:cli",
        ],
    },
)
