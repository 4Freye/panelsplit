# Changelog

All notable changes to panelsplit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - 2025-10-28
### Added
- Consistent type hints with more restrictions (E.g. `--disallow-untyped-defs` `--disallow-incomplete-defs`), addressing [#85](https://github.com/4Freye/panelsplit/issues/85)
- Consistent docstrings addressing [#94](https://github.com/4Freye/panelsplit/issues/94)
- mypy and pydoclint checks on `pre-commit-config.yaml` and `.github/workflows/lint.yml`


## [1.1.1] - 2025-10-23
### Changed
- Migrated from boolean indexing to purely integer-based indexing, as mentioned in [#86](https://github.com/4Freye/panelsplit/issues/86)
### Added
- Consistent type hints throughout the Python codebase, addressing [#85](https://github.com/4Freye/panelsplit/issues/85)
- mypy to CI, addressing [#85](https://github.com/4Freye/panelsplit/issues/85)

## [1.1.0] - 2025-10-21
### Added
- Support for more DataFrame types (e.g. polars) via narwhals

## [1.0.4] - 2025-10-16
### Added
- `CHANGELOG.md` - marking changes to the project
- Automation of publishing to pypi
- Dynamic versioning
- Automation of GitHub Releases via `CHANGELOG.md`
