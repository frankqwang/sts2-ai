$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
python "$repoRoot\STS2AI\Python\evaluate_ai.py" @args
