# Managing the PathwayPS package

This is added as git submodule to the main repository. To setup the repository with the submodule, use the following command:
```bash
git submodule init
git submodule update
```

After the setup, to update the submodule to the latest version, use the following command:
```bash
git submodule update --remote
```

Any changes to the submodule needs to be pushed from within the submodule directory.

# Using poetry to manage the package

The basic packages are kept in the default group which can be installed using 
```bash 
poetry install
```
Adding a new package to the default group can be done using 
```bash
poetry add <package-name>
```

## Using groups in poetry

For experimentation and testing, new groups can be create within the pyproject.toml file. For reference, I have added evaluation group for testing purposes. The packages in the evaluation group can be installed using. For groups deemed necesseary you can set optional as false in the pyproject.toml file.
```bash
poetry install --with evaluations
```
To add dependencies to the evaluation group, use the following command
```bash
poetry add pytest --group evaluations
```
For more information on groups, refer to the [official documentation](https://python-poetry.org/docs/managing-dependencies/)