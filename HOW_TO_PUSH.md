# Pushing this project to GitHub

Git is **not installed** on this machine, so this can't be done automatically — but it's
~5 minutes. The `.gitignore` is already set up (it excludes the two friends' repos and
Python junk so only *your* work is published).

## 1. Install Git (once)
Open PowerShell and run:
```powershell
winget install --id Git.Git -e
```
Then **close and reopen** the terminal (so `git` is on PATH). Check it:
```powershell
git --version
```
(If `winget` isn't available, download from https://git-scm.com/download/win and install with defaults.)

## 2. Create an empty repo on GitHub (once)
- Go to https://github.com/new
- Give it a name (e.g. `gru-structural-breaks`).
- **Do NOT** tick "Add a README" / "Add .gitignore" (you already have them).
- Click **Create repository** and copy the URL it shows, e.g.
  `https://github.com/<your-username>/gru-structural-breaks.git`

## 3. First push (once)
In PowerShell, **inside this folder**:
```powershell
git init
git add .
git commit -m "GRU structural-break forecasting coursework"
git branch -M main
git remote add origin https://github.com/<your-username>/gru-structural-breaks.git
git push -u origin main
```
On the first `push`, a browser window opens asking you to sign in to GitHub — approve it
(Git for Windows handles authentication for you; no password typing).

Done — your code is on GitHub.

---

## Updating the repo after you re-run / change things  ← (your question)
**Yes**, this is exactly what Git is for. After re-running the notebook or editing any file,
just run these three commands and the GitHub copy updates:
```powershell
git add .
git commit -m "rerun: updated results and figures"
git push
```
That's the whole loop: **change → add → commit → push**. You can do it as often as you like;
each push overwrites the files on GitHub with your latest versions and keeps a history of every
commit (so you can always go back).

Tip: the helper script `update_github.ps1` does those three commands in one go — run
`./update_github.ps1 "your message"`.

### Notes
- The executed notebook (~1.7 MB with embedded figures) and `figures/` + `results/` are all
  well under GitHub's limits — no Git LFS needed.
- If you'd rather **not** publish the executed outputs, run *Cell → All Outputs → Clear* before
  committing, but keeping them is fine and lets people view results without running anything.
- The friends' repos (`_friend_repo/`, `_friend2_repo/`) are git-ignored on purpose — they are
  other people's code and shouldn't go on your repo.
