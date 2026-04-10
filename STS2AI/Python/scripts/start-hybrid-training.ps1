$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
python "$repoRoot\STS2AI\Python\train_hybrid.py" @args
