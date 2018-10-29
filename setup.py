from setuptools import setup, find_packages

def load_requirements():
    with open('requirements.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    return content

requirements = load_requirements()

setup(
    name='movie_recommendation',
    version='0.1',
    author='sarah',
    author_email='',
    description='Recommendation system to predict preferences of users towards movies',
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=[
    ]
)
