"""
Run complete churn prediction pipeline.
"""

import subprocess

steps = [
    ("Generate Dataset", "python -m src.data_loader"),
    ("Train Models", "python -m src.train"),
    ("Generate SHAP Reports", "python -m src.explain"),
]

for name, command in steps:

    print(f"\n{name}")
    print("-" * 50)

    result = subprocess.run(
        command,
        shell=True,
    )

    if result.returncode != 0:

        print(f"\nFailed during: {name}")

        break

print("\nPipeline execution complete.")
