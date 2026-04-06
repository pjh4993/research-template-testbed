# research-template-testbed

Sample Tier 2 research project for validating the script-runner workflow template.

## Local testing

```bash
uv sync
uv run python -m sample_training.train --max-steps 3
```

## Submit via MCP

> "Run my training at https://github.com/pjh4993/research-template-testbed, branch main,
> entry point sample_training.train, base image trl-cpu"
