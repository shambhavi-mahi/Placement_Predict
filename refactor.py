import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.startswith('def _get_numeric_cols') or line.startswith('def _apply_minmax') or line.startswith('def _apply_standard') or line.startswith('def _apply_robust') or line.startswith('def _apply_encoding'):
        skip = True
    elif skip and line.startswith('# ---------------------------------------------------------------------------'):
        skip = False
        new_lines.append('from models.feature_engg import _apply_minmax, _apply_standard, _apply_robust, _apply_encoding\n')
        new_lines.append(line)
        continue
    elif line.startswith('# Regression Helpers'):
        skip = True
        new_lines.append('from models.regression import _run_multilinear_regression, _run_simple_regression, _run_regularization_models, _run_logistic_regression, MLR_FEATURES\n')
        new_lines.append('from models.decision_tree import _run_tree_models\n\n')
    elif skip and line.startswith('def _get_regression_data'):
        skip = False
        new_lines.append(line)
        continue
    elif skip and line.startswith('@app.route("/regression")'):
        skip = False
        new_lines.append(line)
        continue
    
    if not skip:
        # Also remove Logistic Regression (sklearn) block
        if line.startswith('# Logistic Regression (sklearn'):
            # The # ------- before it is not caught, but we can just ignore
            skip = True
        else:
            new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
