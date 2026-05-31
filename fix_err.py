"""Replace {e} in retry error prints with a format that includes cause chain."""

import re

files = [
    ('core/text_engine/deepseek.py', 61, 62),
    ('core/text_engine/nvidia.py', 54, 55),
    ('core/text_engine/kimi.py', 111, 112),
    ('core/text_engine/gemini.py', 35, 36),
    ('core/asr_engine.py', 86, 87),
]

for path, old_line_idx, new_line_idx in files:
    lines = open(path, encoding='utf-8').readlines()
    old_line = lines[old_line_idx]
    # Get the indentation
    indent = old_line[:len(old_line) - len(old_line.lstrip())]
    
    # Replace the f-string using concatenation instead to avoid nested quote issues
    # Old format: f"...: {e}"
    # New format: f"...: {(repr(e) + chr(10) + repr(e.__cause__)) if e.__cause__ else repr(e)}"
    # But with quote issues... use a different approach
    
    # Simplest: format outside the f-string
    new_lines = [
        f'{indent}_err_msg = repr(e)\n',
        f'{indent}if e.__cause__:\n',
        f'{indent}    _err_msg += "\\n  └─ cause: " + repr(e.__cause__)\n',
    ]
    
    # Replace the old line with the new multi-line block
    lines[old_line_idx] = ''.join(new_lines)
    
    # On the next line, replace the f-string
    for i in range(old_line_idx + 1, len(lines)):
        if 'repr(e)' not in lines[i]:
            # Find the print line and change {e} or {repr(e)} to {_err_msg}
            if '{e}' in lines[i]:
                lines[i] = lines[i].replace('{e}', '{_err_msg}')
            elif '{repr(e)}' in lines[i]:
                lines[i] = lines[i].replace('{repr(e)}', '{_err_msg}')
            break
    
    open(path, 'w', encoding='utf-8').write(''.join(lines))
    print(f'  ✓ {path}')

# Verify all compile
for path, _, _ in files:
    try:
        compile(open(path, encoding='utf-8').read(), path, 'exec')
        print(f'    compile OK')
    except SyntaxError as e:
        print(f'    compile FAIL: {e}')
