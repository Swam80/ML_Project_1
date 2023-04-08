from setuptools import find_packages,setup
from typing import List


## TO GET REQUIRED MODULES AS A LIST TO BE SUPPLIED IN THE SETUP ( Install requires)


Hyphen_e_dot = '-e.'
def get_requirements(file_path:str) -> List[str]:
    '''
    this function will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj :
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
    if Hyphen_e_dot in requirements:
        requirements.remove(Hyphen_e_dot)
    return requirements

setup(
    name = 'MLprojects',
    version= '0.0.1',
    author = 'Swamesh',
    author_email= 'swamesh80@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')


)