from setuptools import setup

setup(
    name='babyai_text',
    version='0.1.0',
    keywords='babyai, text environment',
    description='A text-only extension of BabyAI',
    packages=['babyai_text', 'babyai_text.levels'],
    install_requires=[
        'colorama',
        'termcolor'
    ]
)