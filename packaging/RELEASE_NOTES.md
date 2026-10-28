## Установка

Python и другие программы не нужны: всё необходимое уже внутри приложения. После
запуска в браузере откроется вкладка dustppd. Закройте вкладку или нажмите «Выйти»,
и программа завершится.

| система | файл | как установить |
|---|---|---|
| macOS (Apple Silicon: M1 и новее) | `dustppd-…-macos-arm64.dmg` | откройте образ и перетащите dustppd в «Программы» |
| Windows 10/11 (64 бит) | `dustppd-…-windows-x64-setup.exe` | запустите установщик; права администратора не нужны |
| Linux (x86-64) | `dustppd-…-linux-x86_64.AppImage` | `chmod +x dustppd-*.AppImage` и запустите |

Приложения не подписаны сертификатами Apple и Microsoft, поэтому при первом запуске
система предупредит:

- **macOS:** «Не удалось проверить разработчика». Откройте «Системные настройки →
  Конфиденциальность и безопасность» и нажмите «Всё равно открыть». Это нужно сделать
  один раз. Другой способ — в Терминале: `xattr -dr com.apple.quarantine /Applications/dustppd.app`.
- **Windows:** SmartScreen: «Подробнее → Выполнить в любом случае».
- **Linux:** если AppImage не запускается, установите `libfuse2`
  (`sudo apt install libfuse2`) или используйте архив `.tar.gz` (запуск: `dustppd/dustppd`).

Для Mac с процессором Intel и для работы из Python: `pip install` из исходников (см. README).

---

## Installation

No Python or other software is needed. After launch, a dustppd tab opens in the
browser. Close the tab or press "Quit" to stop the program. The apps are not signed:
on macOS allow them in System Settings → Privacy & Security ("Open Anyway"); on Windows
choose "More info → Run anyway" in SmartScreen.
