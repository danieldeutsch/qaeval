import setuptools

setuptools.setup(
    name='qaeval',
    version='0.0.3',
    author='Daniel Deutsch',
    description='A package for evaluating the content of summaries through question-answering',
    url='https://github.com/danieldeutsch/qaeval',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'allennlp==1.1.0',
        'torch==1.6.0',
        'transformers==3.0.2',
        'urllib3>=1.25.10'
    ]
)