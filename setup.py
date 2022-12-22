from setuptools import setup, find_packages

setup(
  name = 'RPQ-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Reverse Product Quantization (RPQ) of weights to reduce static memory usage.',
  author = 'Ali Kore',
  author_email = 'akore654@gmail.com',
  long_description=open('README.md', 'r').read(),
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/a-kore/RPQ-pytorch',
  keywords = [
    'artificial intelligence',
    'AI',
    'machine learning',
    'deep learning',
    'pytorch',
    'quantization',
    'product quantization',
    'reverse product quantization',
    'memory reduction',
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.6',
    'transformers>=4.0',
    'vit-pytorch>=0.40',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)