# Trading Bot 2.0

Futures trading bot with ML-based price direction prediction using neural networks.

## Project Structure

```
├── AGENTS.md              # Operational guide (build/run/test commands)
├── IMPLEMENTATION_PLAN.md # Prioritized task list (managed by Ralph)
├── PROMPT_build.md        # Build mode instructions
├── PROMPT_plan.md         # Plan mode instructions
├── loop.sh                # Ralph autonomous loop script
├── specs/                 # Requirements specifications
└── src/
    └── ml/                # ML pipeline
        ├── data/          # Data loading, feature engineering
        ├── models/        # Neural network architectures
        ├── utils/         # Evaluation, visualization
        └── configs/       # Configuration files
```

## Tech Stack

- **Language**: Python 3.10+
- **ML Framework**: PyTorch
- **Data**: Pandas, NumPy
- **Validation**: scikit-learn (walk-forward CV)
- **Visualization**: Matplotlib, Seaborn

## Key Commands

```bash
# Install dependencies
pip install -r src/ml/requirements.txt

# Run training
python src/ml/train_futures_model.py --data /path/to/data.txt

# Run tests
pytest src/ml/

# Run linting
ruff check src/ml/
```

## Ralph Workflow

This project uses the RALPH methodology for autonomous development:

1. **Plan Mode**: `./loop.sh plan` - Analyzes specs vs code, generates task list
2. **Build Mode**: `./loop.sh` - Implements tasks autonomously

### Important Files for Ralph

- `specs/*.md` - Source of truth for requirements
- `AGENTS.md` - Operational commands (kept brief, ~60 lines)
- `IMPLEMENTATION_PLAN.md` - Task tracking (auto-managed)

## Development Guidelines

- Use subagents for parallel file operations
- Run validation (tests, typecheck, lint) after changes
- Commit after each completed task
- Update IMPLEMENTATION_PLAN.md with discoveries
- Keep AGENTS.md operational only (no status/progress)
