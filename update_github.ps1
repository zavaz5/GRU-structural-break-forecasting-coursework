# Push your latest changes to GitHub in one command.
# Usage:   ./update_github.ps1 "what changed"
#   e.g.   ./update_github.ps1 "rerun with metric grid"
param([string]$message = "update results")

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. See HOW_TO_PUSH.md (step 1)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path .git)) {
    Write-Host "This folder isn't a git repo yet. Do the one-time setup in HOW_TO_PUSH.md (steps 2-3) first." -ForegroundColor Yellow
    exit 1
}

git add .
git commit -m $message
git push
Write-Host "Pushed: '$message'" -ForegroundColor Green
