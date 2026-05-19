# Deployment Workflow

## Remotes
| Remote   | Destination                                          | Purpose              |
|----------|------------------------------------------------------|----------------------|
| `origin` | https://github.com/TomasettiL/ComicVision.git        | Private source repo  |
| `space`  | https://huggingface.co/spaces/lntom/comicvision      | Public deployment    |

## Everyday workflow
1. Make and test changes locally: `python3 app.py`
2. Commit your changes
3. Push to GitHub first (source of truth)
4. Push to HF Space (triggers an automatic rebuild)

```bash
git add <files>
git commit -m "description of change"
git push origin main      # → GitHub
git push space main       # → Hugging Face (rebuilds in ~2-3 min)
```

## Live URL
https://huggingface.co/spaces/lntom/comicvision

## Notes
- Free tier spaces pause after 48 hrs of no traffic — auto-wake on visit (~30s cold start)
- Every push to `space main` triggers a full Docker rebuild
- Rebuilds are fast after the first one — pip layer is cached unless requirements.txt changes
- `ALLOW_VIDEO` is not set on HF — image-only for public users
- To enable video on HF: Space page → Settings → Variables and Secrets → add `ALLOW_VIDEO = 1`
- Gunicorn is set to `--workers 1` on purpose — the job state lives in memory and is not shared across processes
