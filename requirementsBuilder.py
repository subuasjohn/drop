unset PYTHONPATH
unset PYTHONHOME

# in .bashrc
alias pyClear='unset PYTHONPATH'

## venv here and then:

filename = "watcher.py"
with open(filename) as f:
    code = compile(f.read(), filename, "exec")
    exec(code, globals(), locals())
