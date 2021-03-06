# ProceZeus

[![Travis](https://img.shields.io/travis/Cyberjusticelab/JusticeAI.svg)](https://travis-ci.org/Cyberjusticelab/JusticeAI/) [![Codecov](https://img.shields.io/codecov/c/github/codecov/example-python.svg)](https://codecov.io/gh/Cyberjusticelab/JusticeAI)

## Getting Started

### Prerequisites

All of the project's services are split into separate Docker images. All application dependencies are contained within the Docker images. The dependencies required to run this project locally are:

- `docker`
- `docker-compose`

### Installing

To install Docker and Docker Compose, you can follow the instructions [here](https://docs.docker.com/).

## Running the Entire Application Stack

We've developed a script to help with running the entire application with all its components. All you need is:

```bash
./cjl up
```

If you want to suppress output push the job to the background:

```bash
./cjl up -d
```

Docker isn't always the best at determining diffs between images. You can manually rebuild all images with:

```bash
./cjl build
```

If that doesn't work, try destroying all Docker containers/images on your machine:

```bash
./cjl clean
```

To run all tests and lints for all services:

```bash
./cjl test
```

To try to fix all linting errors for all services:

```bash
./cjl lint-fix
```

In order to shut down all containers:

```bash
./cjl down
```

The `cjl` script also takes any other command that `docker-compose` can take.

## Running or Testing Specific Services

The following services can run individually or tested against:
- [Web Client](src/web_client/README.md)
- [Backend Service](src/backend_service/README.md)
- [Machine Learning Service](src/ml_service/README.md)
- [Natural Language Processing Service](src/nlp_service/README.md)
- [PostgreSQL Database](src/postgresql_db/README.md)

## Deployment

We intend to deploy our application with continuous delivery. This is a task is expected to be completed by Iteration #2.

## Architecture

The following architecture diagram represents the various services and the relationships they have with one another.

![High Level Architecture](/images/high-level-architecture.png)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Versioning

There are currently no releases versions of our software.

## Authors

TBD

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

TBD
