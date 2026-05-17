import nbformat, os

with open('notebooks/capstone_main.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
md_count = sum(1 for c in nb.cells if c.cell_type == 'markdown')
code_count = sum(1 for c in nb.cells if c.cell_type == 'code')
comment_lines = 0
for c in nb.cells:
    if c.cell_type == 'code':
        for line in c.source.split('\n'):
            if line.strip().startswith('#'):
                comment_lines += 1
print(f'Notebook: {len(nb.cells)} cells ({md_count} markdown, {code_count} code)')
print(f'Comment lines in code cells: {comment_lines}')
print()

print('SUBMISSION FILES:')
print('=' * 60)
files = [
    ('notebooks/capstone_main.ipynb', 'Jupyter Notebook'),
    ('REPORT.md', '10-page report'),
    ('outputs/task1_screening.csv', 'Stock screening'),
    ('outputs/task1_selection_justified.json', 'Selection rationale'),
    ('outputs/task3_metrics.json', 'Model metrics'),
    ('outputs/task4_garch.json', 'GARCH volatility'),
    ('outputs/task4_trends.json', 'Trend analysis'),
    ('outputs/task5_allocation.csv', 'Portfolio allocation'),
    ('outputs/task6_comparison.csv', 'Model comparison'),
    ('outputs/task8_actuals.json', 'StockGro actuals'),
    ('outputs/forecasts/all_forecasts.json', 'All forecasts'),
    ('dashboard/app.py', 'Dashboard (bonus)'),
]
for path, desc in files:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = 'OK' if exists else 'MISSING'
    print(f'  [{status}] {path:<50} ({size:,} bytes)')

fig_count = len(os.listdir('outputs/figures'))
ss_count = len(os.listdir('results'))
print(f'\nFigures: {fig_count} plots in outputs/figures/')
print(f'Screenshots: {ss_count} StockGro screenshots in results/')
