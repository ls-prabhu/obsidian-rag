#!/usr/bin/env python3
"""Onboarding UI — pick Obsidian vault using the OS-native file dialog."""

import os
import platform
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv, set_key

load_dotenv()

ENV_PATH = Path(__file__).parent / ".env"
VAULT_KEY = "VAULT_PATH"


def _macos_dialog() -> Path | None:
    """macOS native folder picker via AppleScript (always available)."""
    script = (
        'tell application "Finder" to choose folder '
        'with prompt "Select your Obsidian vault folder"'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            path = result.stdout.strip().rstrip(":")
            return Path(path) if path else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _windows_dialog() -> Path | None:
    """Windows native folder picker via ctypes (pure Python, no deps)."""
    try:
        import ctypes
        from ctypes import wintypes

        # SHBrowseForFolder
        BIF_NEWDIALOGSTYLE = 0x0040
        BIF_EDITBOX = 0x0010
        BIF_RETURNONLYFSDIRS = 0x0001

        shell32 = ctypes.windll.shell32
        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)

        buf = ctypes.create_unicode_buffer(260)

        class BROWSEINFO(ctypes.Structure):
            _fields_ = [
                ("hwndOwner", wintypes.HWND),
                ("pidlRoot", ctypes.c_void_p),
                ("pszDisplayName", wintypes.LPWSTR),
                ("lpszTitle", wintypes.LPCWSTR),
                ("ulFlags", wintypes.UINT),
                ("lpfn", ctypes.c_void_p),
                ("lParam", ctypes.c_void_p),
                ("iImage", ctypes.c_int),
            ]

        bi = BROWSEINFO()
        bi.lpszTitle = "Select your Obsidian vault folder"
        bi.ulFlags = BIF_NEWDIALOGSTYLE | BIF_EDITBOX | BIF_RETURNONLYFSDIRS
        bi.pszDisplayName = buf

        pidl = shell32.SHBrowseForFolder(ctypes.byref(bi))
        if pidl and shell32.SHGetPathFromIDListW(pidl, buf):
            ole32.CoUninitialize()
            return Path(buf.value)

        ole32.CoUninitialize()
    except Exception:
        pass
    return None


def _linux_native_dialog() -> Path | None:
    """Linux native folder picker — try common backends."""
    methods = []

    for exe in ("zenity", "kdialog"):
        if _which(exe):
            methods.append(exe)

    # zenity (GNOME/GTK)
    if "zenity" in methods:
        try:
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory",
                 "--title", "Select your Obsidian vault folder"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # kdialog (KDE/Qt)
    if "kdialog" in methods:
        try:
            result = subprocess.run(
                ["kdialog", "--getexistingdirectory",
                 "--title", "Select your Obsidian vault folder"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # tkinter fallback
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select your Obsidian vault folder")
        root.destroy()
        if folder:
            return Path(folder)
    except Exception:
        pass

    return None


def _which(name: str) -> bool:
    try:
        subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def pick_vault() -> Path | None:
    system = platform.system()
    if system == "Darwin":
        return _macos_dialog()
    elif system == "Windows":
        return _windows_dialog()
    else:
        return _linux_native_dialog()


def prompt_manual() -> Path:
    print("  Install 'zenity' or 'tk' for a GUI picker.")
    while True:
        raw = input("  Enter full path to your Obsidian vault: ").strip()
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
        print(f"  Not a valid directory: {p}")


def _run_step(description: str, *args: str) -> bool:
    print(f"\n  {description}...", end="", flush=True)
    result = subprocess.run(
        [sys.executable, *args],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(" done")
        return True
    print(" FAILED")
    print(result.stderr[:2000] if result.stderr else result.stdout[:2000])
    return False


REQUIRED_KEYS = {
    "GROQ_API_KEY": "Groq API key (router: llama-3.1-8b-instant, RAG: llama-3.3-70b-versatile)",
}


def _ensure_api_keys() -> None:
    """Prompt for any missing required API keys and save them to both .env files."""
    env_files = [
        ("root", Path(__file__).parent / ".env"),
        ("agent", Path(__file__).parent / "obsidian" / ".env"),
    ]

    for filename, env_path in env_files:
        load_dotenv(env_path)

    for key, description in REQUIRED_KEYS.items():
        val = os.getenv(key, "")
        if val and not val.startswith("sk-...") and not val.startswith("AIzaSy"):
            continue
        raw = input(f"\n  {description}\n  {key}: ").strip()
        if raw:
            for _, env_path in env_files:
                set_key(str(env_path), key, raw)
            os.environ[key] = raw


def main():
    existing_vault = os.getenv("VAULT_PATH", "")
    if existing_vault:
        p = Path(existing_vault).expanduser()
        if p.is_dir():
            print(f"Using existing VAULT_PATH={p}")
            vault = p
        else:
            print(f"Existing VAULT_PATH={p} is not a valid directory, selecting new one...")
            vault = pick_vault()
            if vault is None:
                vault = prompt_manual()
    else:
        print("Selecting Obsidian vault folder...")
        vault = pick_vault()
        if vault is None:
            vault = prompt_manual()

    vault_str = str(vault.resolve())
    set_key(str(ENV_PATH), VAULT_KEY, vault_str)
    print(f"\nSaved VAULT_PATH={vault_str} to {ENV_PATH}")

    _ensure_api_keys()

    steps_dir = Path(__file__).parent
    steps = [
        ("Reading vault", steps_dir / "read.py"),
        ("Chunking notes", steps_dir / "chunk.py"),
        ("Building embeddings", steps_dir / "embeddings.py"),
    ]

    for desc, script in steps:
        if not _run_step(desc, str(script)):
            sys.exit(1)

    print("\nLaunching ADK agent...")
    subprocess.run(["adk", "run", "obsidian"])


if __name__ == "__main__":
    main()
