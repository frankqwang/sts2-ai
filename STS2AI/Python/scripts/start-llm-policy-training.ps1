$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
python "$repoRoot\STS2AI\Python\train_llm_policy.py" @args
