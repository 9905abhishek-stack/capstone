import json
with open('stock_price_prediction_using_yfinance[1].ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source_lines = cell.get('source', [])
    if not source_lines: continue
    
    first_line = source_lines[0].strip()
    if ctype == 'markdown':
        if first_line.startswith('#'):
            print(f"Cell {i} (MD): {first_line}")
        else:
            print(f"Cell {i} (MD): {first_line[:50]}...")
    elif ctype == 'code':
        print(f"Cell {i} (Code, {len(source_lines)} lines): {first_line[:50]}...")
