# (WORK IN PROGRESS) FastTags + DataStar Test Implementation

This is a minimal implementation of FastHTML leveraging Datastar for DOM updates and signal management


## to see demo

1. clone
2. 
```zsh
uvx marimo edit --sandbox notebooks/core_ds.py
```

## 🧪 Testing

To test the export process, run `.github/scripts/build.py` from the root directory.

```bash
uv run .github/scripts/build.py
```

This will export all notebooks in a folder called `_site/` in the root directory. Then to serve the site, run:

```bash
python -m http.server -d _site
```

This will serve the site at `http://localhost:8000`.

[] TODO clean up build and look at |# export.. 
[] Ngrok implementation for local host. 