# 🔧 Devcontainer Setup für BA-Predictive-Maintenance-SCANIA

Dieses Dokument beschreibt die initiale Einrichtung des Entwicklungscontainers sowie die Konfiguration von Git und Python-Abhängigkeiten für dieses Projekt.

---

## Voraussetzungen

Bevor du den Devcontainer verwendest, solltest du:

1. Ein GitHub-Repository erstellt haben:  
   `https://github.com/JustinStange-Heiduk/BA-Predictive-Maintenance-SCANIA`

2. Dieses Repository **per HTTPS geklont** haben (nicht via SSH), z. B. mit:
   ```bash
   git clone https://github.com/JustinStange-Heiduk/BA-Predictive-Maintenance-SCANIA.git
   ```

3. VS Code mit folgenden Erweiterungen installiert haben:
   - **Remote – Containers**
   - **GitHub Codespaces** (optional, für GitHub-Integration)
   - **Python Extension Pack**

---

## Schritt-für-Schritt Setup

### 1. Projekt in VS Code öffnen

Nach dem Klonen öffne den Ordner in Visual Studio Code. Wenn du gefragt wirst:

> **„Reopen in Container?“** → **Klicke auf „Ja“**

Alternativ:  
Öffne die Command Palette (`Strg+Shift+P`) und wähle:  
> `Dev Containers: Reopen in Container`

---

### 2. Git-Konfiguration im Container

Die globale Git-Konfiguration wird automatisch beim ersten Start gesetzt – durch diesen Befehl in `.devcontainer/devcontainer.json`:

```json
"postCreateCommand": "git config --global user.name '<>' && git config --global user.email '<>' && pip install -r requirements.txt"
```

Damit sind dein Name und deine E-Mail für Commits korrekt hinterlegt, und alle Python-Abhängigkeiten aus `requirements.txt` werden installiert.

---

### 3. GitHub-Remote setzen (nur beim ersten Mal nötig)

Falls dein Container noch nicht mit dem richtigen Remote verknüpft ist, setze ihn manuell:

```bash
git remote set-url origin https://github.com/JustinStange-Heiduk/BA-Predictive-Maintenance-SCANIA.git
```

Überprüfe mit:

```bash
git remote -v
```

Du solltest Folgendes sehen:

```
origin  https://github.com/JustinStange-Heiduk/BA-Predictive-Maintenance-SCANIA.git (fetch)
origin  https://github.com/JustinStange-Heiduk/BA-Predictive-Maintenance-SCANIA.git (push)
```

---

### 4. Push mit Personal Access Token (PAT)

Beim ersten Push-Versuch wirst du nach deinem GitHub-Benutzernamen und Passwort gefragt. Gib ein:

- **Username:** `JustinStange-Heiduk`
- **Passwort:** dein [GitHub Personal Access Token (PAT)](https://github.com/settings/tokens)

#### Token-Speicherung für spätere Pushes

Im Container kannst du deinen Token dauerhaft speichern mit:

```bash
git config --global credential.helper store
```

Achtung: Der Token wird dann unverschlüsselt in `~/.git-credentials` gespeichert (nur innerhalb des Containers sichtbar).

---