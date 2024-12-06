# Project Setup

## 1. Create the Virtual Environment

If you haven't created the virtual environment yet, do so by running the following command:

- On **Windows** (PowerShell):
  ```bash
  python -m venv qiskitEnv
  ```
- On **Windows** (Command Prompt):
  ```bash
  python -m venv qiskitEnv
  ```
- On **macOS/Linux**:
  ```bash
  python3 -m venv qiskitEnv
  ```

This will create a folder named `qiskitEnv` with the virtual environment files.

## 2. Activate the Virtual Environment

To activate the virtual environment, use the following command:

- On **Windows** (PowerShell):
  ```bash
  .qiskitEnv\Scripts\Activate
  ```
- On **Windows** (Command Prompt):
  ```bash
  .qiskitEnv\Scripts\activate.bat
  ```
- On **macOS/Linux**:
  ```bash
  source qiskitEnv/bin/activate
  ```

Once activated, your terminal prompt should show `(qiskitEnv)`.

## 3. `requirements.txt`

To generate the `requirements.txt` file with all currently installed dependencies:

```bash
pip freeze > requirements.txt
```

To install the dependencies listed in `requirements.txt`, use the following command:

```bash
pip install -r requirements.txt
```
```