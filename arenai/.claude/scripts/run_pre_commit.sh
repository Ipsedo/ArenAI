if [ -x .venv/bin/pre-commit ]; then PC=.venv/bin/pre-commit; else PC=pre-commit; fi
command -v "$PC" >/dev/null 2>&1 || exit 0
i=0
until "$PC" run --all-files; do    # rejoue tant que pre-commit échoue/modifie
  i=$((i+1))
  [ "$i" -ge 5 ] && break          # garde-fou : max 5 passages
done
true                               # sort toujours en 0 → ne bloque jamais le tour
