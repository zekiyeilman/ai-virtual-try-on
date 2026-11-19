"""PreToolUse kancası: geri alınamaz git komutlarını Claude çalıştırmadan önce engeller.

git-guardrails-claude-code skill'inin Python sürümü (Windows'ta jq yok). Bash ve PowerShell
araçlarının `tool_input.command` alanını kontrol eder; eşleşmede çıkış kodu 2 ile engeller.
Normal `git push` serbesttir; yalnız force push engellenir (kullanıcı tercihi, 2026-11-19).
"""
import json
import re
import sys

# Tek komut parçası içinde arar (; & | satır sonu sınırı aşılmaz) → yanlış pozitif azalır.
_S = "[^;&|\\n]*"
DANGEROUS = [
    (r"git\s+push\b" + _S + r"(--force|--force-with-lease|\s-f\b|\s\+\S)", "force push"),
    (r"git\s+reset\b" + _S + r"--hard", "reset --hard"),
    (r"git\s+clean\b" + _S + r"\s-[a-zA-Z]*f", "clean -f"),
    (r"git\s+branch\b" + _S + r"\s-D\b", "branch -D"),
    (r"git\s+checkout\s+(--\s+)?\.(\s|$)", "checkout ."),
    (r"git\s+restore\b" + _S + r"\s\.(\s|$)", "restore ."),
    (r"git\s+filter-branch|git\s+filter-repo", "geçmiş yeniden yazma"),
]


def main() -> int:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        command = json.load(sys.stdin).get("tool_input", {}).get("command", "") or ""
    except (ValueError, AttributeError):
        return 0
    for pattern, name in DANGEROUS:
        if re.search(pattern, command):
            sys.stderr.write(
                f"BLOCKED: '{command}' tehlikeli git komutu ({name}). "
                "Kullanıcı bunu engelledi; gerekiyorsa kullanıcıdan kendisinin çalıştırmasını iste.\n"
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
