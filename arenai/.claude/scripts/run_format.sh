if [ -x .venv/bin/pre-commit ]; then PC=.venv/bin/pre-commit; else PC=pre-commit; fi

command -v "$PC" >/dev/null 2>&1 || exit 0

i=0

until "$PC" run end-of-file-fixer --all-files \
    && "$PC" run trailing-whitespace --all-files \
    && "$PC" run clang-format --all-files; do
  i=$((i+1))
  [ "$i" -ge 5 ] && break
done
true
