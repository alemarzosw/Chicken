#!/bin/sh
# Install Chicken: a private Python environment next to agent.py, and the `chicken` and `call` commands in ~/.local/bin
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
mkdir -p "$HOME/.local/bin"

cat > "$HOME/.local/bin/chicken" <<EOS
#!/bin/sh
# chicken [options]  — open Chicken, the local AI agent
exec "$DIR/.venv/bin/python" "$DIR/agent.py" "\$@"
EOS

cat > "$HOME/.local/bin/call" <<'EOS'
#!/bin/sh
# call chicken [options]  — open Chicken, the local AI agent
# (inside Chicken, `call -agent <helper> <task>` runs a helper)
case "$1" in
  chicken|-chicken|--chicken) shift; exec "$HOME/.local/bin/chicken" "$@" ;;
  -agent|--agent)
     printf '\033[1;38;2;250;200;30mcall chicken\033[0m opens Chicken. \033[1;38;2;255;95;215mcall -agent <helper> <task>\033[0m is for inside Chicken, to run a helper.\n' >&2
     exit 1 ;;
  *) printf '\033[1;38;2;250;200;30mcall chicken\033[0m [options]   open Chicken, the local AI agent\n' >&2
     printf 'e.g.  call chicken · call chicken "fix the typo in index.html" · call chicken -l "<goal>"\n' >&2
     exit 1 ;;
esac
EOS

chmod +x "$HOME/.local/bin/chicken" "$HOME/.local/bin/call"
echo "Installed. Run 'chicken' (or 'call chicken') in the folder you want to work in."
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "Add ~/.local/bin to your PATH first." ;; esac
