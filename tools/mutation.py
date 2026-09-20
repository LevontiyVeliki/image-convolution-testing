"""Мутационное тестирование набора тестов.

Покрытие показывает, какие строки выполнялись, но не то, проверяют
ли тесты их поведение. Мутационное тестирование отвечает на второй
вопрос: в исходный код вносится небольшая ошибка, после чего
запускаются тесты. Если хотя бы один тест падает, мутация
«уничтожена» — набор заметил подмену. Если все проходят, мутация
«выжила» и указывает на пробел в проверках.

Доля уничтоженных мутаций называется мутационным счётом.

Запуск:

    python tools/mutation.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Вносимые изменения: файл, исходный текст, заменяющий текст,
#: краткое описание внесённой ошибки
MUTATIONS = [
    ("src/imgconv/borders.py",
     "return folded if folded < length else period - folded",
     "return folded if folded <= length else period - folded",
     "Отражение границ: строгое сравнение заменено нестрогим"),

    ("src/imgconv/borders.py",
     "return min(max(index, 0), length - 1)",
     "return min(max(index, 0), length)",
     "Повтор края: выход за верхнюю границу на единицу"),

    ("src/imgconv/borders.py",
     "period = 2 * length - 2",
     "period = 2 * length - 1",
     "Отражение границ: изменён период"),

    ("src/imgconv/kernel.py",
     "if rows % 2 == 0:",
     "if rows % 2 == 1:",
     "Проверка ядра: чётность заменена на нечётность"),

    ("src/imgconv/kernel.py",
     "return total if total != 0.0 else 1.0",
     "return total if total != 0.0 else 0.0",
     "Выбор делителя: единица заменена нулём"),

    ("src/imgconv/kernel.py",
     "if rows != columns:",
     "if rows < columns:",
     "Проверка ядра: ослаблено условие квадратности"),

    ("src/imgconv/convolution.py",
     "np.clip(result, 0.0, MAX_LEVEL, out=result)",
     "np.clip(result, 0.0, MAX_LEVEL + 1, out=result)",
     "Отсечение диапазона: верхняя граница сдвинута"),

    ("src/imgconv/convolution.py",
     "result = accumulator / kernel.divisor + kernel.offset",
     "result = accumulator / kernel.divisor - kernel.offset",
     "Нормализация: сдвиг вычитается вместо сложения"),

    ("src/imgconv/convolution.py",
     "if array.shape[2] not in (1, 3, 4):",
     "if array.shape[2] not in (1, 2, 3, 4):",
     "Проверка изображения: разрешено недопустимое число каналов"),

    ("src/imgconv/convolution.py",
     "if weight == 0.0:\n                continue",
     "if weight != 0.0:\n                continue",
     "Цикл свёртки: пропускаются ненулевые коэффициенты"),

    ("src/imgconv/convolution.py",
     "result[:, :, 3] = source[:, :, 3].astype(np.uint8)",
     "result[:, :, 3] = 255",
     "Прозрачность: канал заменён непрозрачным"),

    ("src/imgconv/imageio.py",
     'target = "RGBA" if _has_alpha(handle) else "RGB"',
     'target = "RGB"',
     "Чтение файла: прозрачность отбрасывается"),
]


def run_suite(directory: Path) -> bool:
    """Запускает тесты в указанном каталоге.

    Возвращает True, если все тесты прошли.
    """
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=directory, capture_output=True, env=environment,
        timeout=600,
    )
    return completed.returncode == 0


def main() -> int:
    print("Мутационное тестирование")
    print(f"Всего мутаций: {len(MUTATIONS)}\n")

    results = []
    killed = 0

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary) / "project"

        for number, (relative, original, mutated, description) in \
                enumerate(MUTATIONS, start=1):
            if base.exists():
                shutil.rmtree(base)
            shutil.copytree(ROOT, base,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", ".git", ".pytest_cache",
                                "htmlcov", ".venv"))

            target = base / relative
            source = target.read_text(encoding="utf-8")

            if original not in source:
                print(f"{number:>2}. ПРОПУЩЕНА: текст не найден "
                      f"в {relative}")
                results.append({"number": number, "file": relative,
                                "description": description,
                                "status": "пропущена"})
                continue

            target.write_text(source.replace(original, mutated, 1),
                              encoding="utf-8")

            passed = run_suite(base)
            status = "выжила" if passed else "уничтожена"
            if not passed:
                killed += 1

            print(f"{number:>2}. {status:<11} {description}")
            results.append({"number": number, "file": relative,
                            "description": description,
                            "status": status})

    considered = [row for row in results if row["status"] != "пропущена"]
    score = killed / len(considered) * 100 if considered else 0.0

    print(f"\nУничтожено {killed} из {len(considered)}")
    print(f"Мутационный счёт: {score:.1f} %")

    report = {"total": len(considered), "killed": killed,
              "score_percent": round(score, 1), "mutations": results}
    path = ROOT / "docs" / "mutation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"Результаты сохранены: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
