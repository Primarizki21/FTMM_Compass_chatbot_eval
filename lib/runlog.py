"""Catat eksekusi skrip: terminal (run.log), event per tahap (log.jsonl), kegagalan (errors.log).

Semua berkas ditulis di direktori run, bukan di root repo, agar satu run bisa diaudit tanpa
menebak perintah atau urutan tahap.
"""

import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def catat_event(directory, **fields) -> None:
    """Satu baris JSON per tahap; dipakai untuk merekonstruksi urutan eksekusi."""
    line = {"waktu": datetime.now().isoformat(timespec="seconds"), **fields}
    with (Path(directory) / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


@contextmanager
def catat_run(directory, argv, nama_log: str = "run.log"):
    """Salin stdout dan stderr ke `<directory>/<nama_log>`; traceback ke `errors.log`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    perintah = " ".join(str(arg) for arg in argv)
    with (directory / nama_log).open("w", encoding="utf-8") as log:
        log.write(f"# mulai: {datetime.now().isoformat(timespec='seconds')}\n# argv: {perintah}\n")
        log.flush()
        asli_out, asli_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(asli_out, log), _Tee(asli_err, log)
        try:
            yield
        except BaseException:
            with (directory / "errors.log").open("a", encoding="utf-8") as err:
                err.write(f"# {datetime.now().isoformat(timespec='seconds')} argv: {perintah}\n")
                err.write(traceback.format_exc())
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = asli_out, asli_err
