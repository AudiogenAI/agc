from setuptools import setup, find_packages

setup(
    name="audiogen-agc",
    version="0.1.1",
    author="Elio Pascarelli",
    author_email="elio@audiogen.co",
    description="Audiogen Codec",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AudiogenAI/agc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "torch",
        "torchaudio",
        "transformers",
        "einops",
        "numpy"
    ],
    python_requires=">=3.9",
)
