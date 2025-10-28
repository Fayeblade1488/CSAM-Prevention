# Contributing Guide

## Development Setup

```bash
# Clone and setup
git clone https://github.com/Fayeblade1488/CSAM-Prevention
cd CSAM-Prevention
make setup
make test
```

## Workflow

1. **Fork** the repository
2. **Branch:** Use format `type/scope-description`
   - `feat/add-auth-system`
   - `fix/memory-leak-handler`
   - `docs/api-reference-update`
3. **Commit:** Follow [Conventional Commits](https://www.conventionalcommits.org/)
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `chore:` maintenance tasks
   - `refactor:` code restructuring
   - `test:` test additions/modifications
   - `perf:` performance improvements
4. **Test:** All tests must pass (`make test`)
5. **Lint:** Code must pass linting (`make lint`)
6. **PR:** Open against `main` with clear description
7. **Review:** Address feedback; squash commits before merge

## Code Standards

- **Style:** Follow language conventions (PEP 8)
- **Types:** Use type hints/annotations where supported
- **Tests:** Maintain >80% coverage for new code
- **Documentation:** Update relevant docs for user-facing changes
- **Security:** No secrets in code; scan with `make security-check`

## AI-Assisted Development

- Follow guidelines in [AI_USAGE.md](AI_USAGE.md)
- Disclose AI tool usage in PR descriptions
- Ensure AI-generated code is reviewed and tested

## PR Requirements Checklist

- [ ] Tests pass locally (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation updated for user-facing changes
- [ ] CHANGELOG.md updated (for releases)
- [ ] AI usage disclosed (if applicable)
- [ ] Approved by required CODEOWNERS
- [ ] No merge conflicts with target branch
