"""Menampilkan model Gemini yang bisa dipakai API key di .env untuk menjawab chat."""

import sys

from lib import llm


def chat_model_rows(models) -> list[tuple[str, str, int, int]]:
    rows = []
    for model in models:
        if "generateContent" in (model.supported_actions or []):
            name = model.name.removeprefix("models/")
            rows.append((name, model.display_name or "", model.input_token_limit or 0, model.output_token_limit or 0))
    return sorted(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    problem = llm.gemini_setup_problem()
    if problem:
        print(problem)
        return 1

    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=llm.os.environ["GEMINI_API_KEY"])
    try:
        rows = chat_model_rows(client.models.list())
    except errors.APIError as error:
        print(f"Gemini API menolak permintaan ({error.code}): {error.message}")
        return 1

    print(f"{'NAMA MODEL':<42}{'TOKEN MASUK':>13}{'TOKEN KELUAR':>14}  NAMA TAMPILAN")
    for name, display_name, input_limit, output_limit in rows:
        print(f"{name:<42}{input_limit:>13,}{output_limit:>14,}  {display_name}")
    print(f"\n{len(rows)} model. Salin NAMA MODEL ke kolom Model LLM di sidebar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
