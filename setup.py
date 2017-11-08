from distutils.core import setup

setup(
    name='personalization-theory',
    version='0.1',
    packages=['personalization-theory'],
    url='https://github.com/taeyoung-choi/personalization-theory',
    author='IEORE4571',
    install_requires =[
        'surprise',
        'numpy',
        'matplotlib',
        'pandas'
    ]
)
