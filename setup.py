import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='tf_taggers',
    version='0.0.6',
    packages=['tf_taggers'],
    include_package_data=True,
    license='MIT License',
    description='TextFlows taggers module',
    long_description=README,
    url='https://github.com/xflows/tf_taggers',
    author='Matej Martinc',
    author_email='matej.martinc@ijs.si',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries'
    ],
    install_requires=[
        "nltk==3.4.5",
        "django-discover-runner==1.0",
        "numpy",
        "scipy",
        "scikit-learn==0.16.1",
        "docx",
        "requests",
        "pdfminer",
        "python-crfsuite"
    ]
)
